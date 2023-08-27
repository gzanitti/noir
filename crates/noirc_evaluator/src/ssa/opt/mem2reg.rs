//! The goal of the mem2reg SSA optimization pass is to replace any `Load` instructions to known
//! addresses with the value stored at that address, if it is also known. This pass will also remove
//! any `Store` instructions within a block that are no longer needed because no more loads occur in
//! between the Store in question and the next Store.
//!
//! The pass works as follows:
//! - Each block in each function is iterated in forward-order.
//! - The starting value of each reference in the block is the unification of the same references
//!   at the end of each direct predecessor block to the current block.
//! - At each step, the value of each reference is either Known(ValueId) or Unknown.
//! - Two reference values unify to each other if they are exactly equal, or to Unknown otherwise.
//! - If a block has no predecessors, the starting value of each reference is Unknown.
//! - Throughout this pass, aliases of each reference are also tracked.
//!   - References typically have 1 alias - themselves.
//!   - A reference with multiple aliases means we will not be able to optimize out loads if the
//!     reference is stored to. Note that this means we can still optimize out loads if these
//!     aliased references are never stored to, or the store occurs after a load.
//!   - A reference with 0 aliases means we were unable to find which reference this reference
//!     refers to. If such a reference is stored to, we must conservatively invalidate every
//!     reference in the current block.
//!
//! From there, to figure out the value of each reference at the end of block, iterate each instruction:
//! - On `Instruction::Allocate`:
//!   - Register a new reference was made with itself as its only alias
//! - On `Instruction::Load { address }`:
//!   - If `address` is known to only have a single alias (including itself) and if the value of
//!     that alias is known, replace the value of the load with the known value.
//!   - Furthermore, if the result of the load is a reference, mark the result as an alias
//!     of the reference it dereferences to (if known).
//!     - If which reference it dereferences to is not known, this load result has no aliases.
//! - On `Instruction::Store { address, value }`:
//!   - If the address of the store is `Known(value)`:
//!     - If the address has exactly 1 alias:
//!       - Set the value of the address to `Known(value)`.
//!     - If the address has more than 1 alias:
//!       - Set the value of every possible alias to `Unknown`.
//!     - If the address has 0 aliases:
//!       - Conservatively mark every alias in the block to `Unknown`.
//!   - If the address of the store is not known:
//!     - Conservatively mark every alias in the block to `Unknown`.
//!   - Additionally, if there were no Loads to any alias of the address between this Store and
//!     the previous Store to the same address, the previous store can be removed.
//! - On `Instruction::Call { arguments }`:
//!   - If any argument of the call is a reference, set the value of each alias of that
//!     reference to `Unknown`
//!   - Any builtin functions that may return aliases if their input also contains a
//!     reference should be tracked. Examples: `slice_push_back`, `slice_insert`, `slice_remove`, etc.
//!
//! On a terminator instruction:
//! - If the terminator is a `Jmp`:
//!   - For each reference argument of the jmp, mark the corresponding block parameter it is passed
//!     to as an alias for the jmp argument.
//!
//! Finally, if this is the only block in the function, we can remove any Stores that were not
//! referenced by the terminator instruction.
//!
//! Repeating this algorithm for each block in the function in program order should result in
//! optimizing out most known loads. However, identifying all aliases correctly has been proven
//! undecidable in general (Landi, 1992). So this pass will not always optimize out all loads
//! that could theoretically be optimized out. This pass can be performed at any time in the
//! SSA optimization pipeline, although it will be more successful the simpler the program's CFG is.
//! This pass is currently performed several times to enable other passes - most notably being
//! performed before loop unrolling to try to allow for mutable variables used for loop indices.
use std::collections::{BTreeMap, BTreeSet};

use crate::ssa::{
    ir::{
        basic_block::BasicBlockId,
        cfg::ControlFlowGraph,
        dom::DominatorTree,
        function::Function,
        instruction::{Instruction, InstructionId, TerminatorInstruction},
        post_order::PostOrder,
        types::Type,
        value::ValueId,
    },
    ssa_gen::Ssa,
};

impl Ssa {
    /// Attempts to remove any load instructions that recover values that are already available in
    /// scope, and attempts to remove stores that are subsequently redundant.
    pub(crate) fn mem2reg(mut self) -> Ssa {
        for function in self.functions.values_mut() {
            let mut context = PerFunctionContext::new(function);
            context.mem2reg(function);
            context.remove_instructions(function);
        }
        self
    }
}

struct PerFunctionContext {
    cfg: ControlFlowGraph,
    post_order: PostOrder,
    dom_tree: DominatorTree,

    blocks: BTreeMap<BasicBlockId, Block>,

    /// Load and Store instructions that should be removed at the end of the pass.
    ///
    /// We avoid removing individual instructions as we go since removing elements
    /// from the middle of Vecs many times will be slower than a single call to `retain`.
    instructions_to_remove: BTreeSet<InstructionId>,
}

#[derive(Debug, Default, Clone)]
struct Block {
    /// Maps a ValueId to the Expression it represents.
    /// Multiple ValueIds can map to the same Expression, e.g.
    /// dereferences to the same allocation.
    expressions: BTreeMap<ValueId, Expression>,

    /// Each expression is tracked as to how many aliases it
    /// may have. If there is only 1, we can attempt to optimize
    /// out any known loads to that alias. Note that "alias" here
    /// includes the original reference as well.
    aliases: BTreeMap<Expression, BTreeSet<ValueId>>,

    /// Each allocate instruction result (and some reference block parameters)
    /// will map to a Reference value which tracks whether the last value stored
    /// to the reference is known.
    references: BTreeMap<ValueId, ReferenceValue>,
}

/// An `Expression` here is used to represent a canonical key
/// into the aliases map since otherwise two dereferences of the
/// same address will be given different ValueIds.
#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
enum Expression {
    Dereference(Box<Expression>),
    Other(ValueId),
}

/// Every reference's value is either Known and can be optimized away, or Unknown.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ReferenceValue {
    Unknown,
    Known(ValueId),
}

impl PerFunctionContext {
    fn new(function: &Function) -> Self {
        let cfg = ControlFlowGraph::with_function(function);
        let post_order = PostOrder::with_function(function);
        let dom_tree = DominatorTree::with_cfg_and_post_order(&cfg, &post_order);

        PerFunctionContext {
            cfg,
            post_order,
            dom_tree,
            blocks: BTreeMap::new(),
            instructions_to_remove: BTreeSet::new(),
        }
    }

    /// Apply the mem2reg pass to the given function.
    ///
    /// This function is expected to be the same one that the internal cfg, post_order, and
    /// dom_tree were created from.
    fn mem2reg(&mut self, function: &mut Function) {
        // Iterate each block in reverse post order = forward order
        let mut block_order = PostOrder::with_function(function).into_vec();
        block_order.reverse();

        for block in block_order {
            let references = self.find_starting_references(block);
            self.analyze_block(function, block, references);
        }
    }

    /// The value of each reference at the start of the given block is the unification
    /// of the value of the same reference at the end of its predecessor blocks.
    fn find_starting_references(&mut self, block: BasicBlockId) -> Block {
        let mut predecessors = self.cfg.predecessors(block);

        if let Some(first_predecessor) = predecessors.next() {
            let first = self.blocks.get(&first_predecessor).cloned().unwrap_or_default();

            // Note that we have to start folding with the first block as the accumulator.
            // If we started with an empty block, an empty block union'd with any other block
            // is always also empty so we'd never be able to track any references across blocks.
            predecessors.fold(first, |block, predecessor| {
                let predecessor = self.blocks.entry(predecessor).or_default();
                block.unify(predecessor)
            })
        } else {
            Block::default()
        }
    }

    /// Analyze a block with the given starting reference values.
    ///
    /// This will remove any known loads in the block and track the value of references
    /// as they are stored to. When this function is finished, the value of each reference
    /// at the end of this block will be remembered in `self.blocks`.
    fn analyze_block(
        &mut self,
        function: &mut Function,
        block: BasicBlockId,
        mut references: Block,
    ) {
        let instructions = function.dfg[block].instructions().to_vec();
        let mut last_stores = BTreeMap::new();

        for instruction in instructions {
            self.analyze_instruction(function, &mut references, instruction, &mut last_stores);
        }

        self.handle_terminator(function, block, &mut references, &mut last_stores);

        // If there's only 1 block in the function total, we can remove any remaining last stores
        // as well. We can't do this if there are multiple blocks since subsequent blocks may
        // reference these stores.
        if self.post_order.as_slice().len() == 1 {
            self.remove_stores_that_do_not_alias_parameters(function, &references, last_stores);
        }

        self.blocks.insert(block, references);
    }

    /// Add all instructions in `last_stores` to `self.instructions_to_remove` which do not
    /// possibly alias any parameters of the given function.
    fn remove_stores_that_do_not_alias_parameters(
        &mut self,
        function: &Function,
        references: &Block,
        last_stores: BTreeMap<ValueId, InstructionId>,
    ) {
        let reference_parameters = function
            .parameters()
            .iter()
            .filter(|param| function.dfg.type_of_value(**param) == Type::Reference)
            .collect::<BTreeSet<_>>();

        for (allocation, instruction) in last_stores {
            if let Some(expression) = references.expressions.get(&allocation) {
                if let Some(aliases) = references.aliases.get(expression) {
                    let allocation_aliases_parameter =
                        aliases.iter().any(|alias| reference_parameters.contains(alias));

                    if !aliases.is_empty() && !allocation_aliases_parameter {
                        self.instructions_to_remove.insert(instruction);
                    }
                }
            } else {
                self.instructions_to_remove.insert(instruction);
            }
        }
    }

    fn analyze_instruction(
        &mut self,
        function: &mut Function,
        references: &mut Block,
        instruction: InstructionId,
        last_stores: &mut BTreeMap<ValueId, InstructionId>,
    ) {
        match &function.dfg[instruction] {
            Instruction::Load { address } => {
                let address = function.dfg.resolve(*address);

                let result = function.dfg.instruction_results(instruction)[0];
                references.remember_dereference(function, address, result);

                // If the load is known, replace it with the known value and remove the load
                if let Some(value) = references.get_known_value(address) {
                    let result = function.dfg.instruction_results(instruction)[0];
                    function.dfg.set_value_from_id(result, value);

                    self.instructions_to_remove.insert(instruction);
                } else {
                    last_stores.remove(&address);
                }
            }
            Instruction::Store { address, value } => {
                let address = function.dfg.resolve(*address);
                let value = function.dfg.resolve(*value);

                // If there was another store to this instruction without any (unremoved) loads or
                // function calls in-between, we can remove the previous store.
                if let Some(last_store) = last_stores.get(&address) {
                    self.instructions_to_remove.insert(*last_store);
                }

                references.set_known_value(address, value, last_stores);
                last_stores.insert(address, instruction);
            }
            Instruction::Call { arguments, .. } => {
                self.mark_all_unknown(arguments, function, references, last_stores);
            }
            Instruction::Allocate => {
                // Register the new reference
                let result = function.dfg.instruction_results(instruction)[0];
                references.expressions.insert(result, Expression::Other(result));

                let mut aliases = BTreeSet::new();
                aliases.insert(result);
                references.aliases.insert(Expression::Other(result), aliases);
            }
            _ => (),
        }
    }

    fn mark_all_unknown(
        &self,
        values: &[ValueId],
        function: &Function,
        references: &mut Block,
        last_stores: &mut BTreeMap<ValueId, InstructionId>,
    ) {
        for value in values {
            if function.dfg.type_of_value(*value) == Type::Reference {
                let value = function.dfg.resolve(*value);
                references.set_unknown(value, last_stores);
                last_stores.remove(&value);
            }
        }
    }

    /// Remove any instructions in `self.instructions_to_remove` from the current function.
    /// This is expected to contain any loads which were replaced and any stores which are
    /// no longer needed.
    fn remove_instructions(&self, function: &mut Function) {
        // The order we iterate blocks in is not important
        for block in self.post_order.as_slice() {
            function.dfg[*block]
                .instructions_mut()
                .retain(|instruction| !self.instructions_to_remove.contains(instruction));
        }
    }

    fn handle_terminator(
        &self,
        function: &mut Function,
        block: BasicBlockId,
        references: &mut Block,
        last_stores: &mut BTreeMap<ValueId, InstructionId>,
    ) {
        match function.dfg[block].unwrap_terminator() {
            TerminatorInstruction::JmpIf { .. } => (), // Nothing to do
            TerminatorInstruction::Jmp { destination, arguments, .. } => {
                let destination_parameters = function.dfg[*destination].parameters();
                assert_eq!(destination_parameters.len(), arguments.len());

                // Add an alias for each reference parameter
                for (parameter, argument) in destination_parameters.iter().zip(arguments) {
                    if function.dfg.type_of_value(*parameter) == Type::Reference {
                        let argument = function.dfg.resolve(*argument);

                        if let Some(expression) = references.expressions.get(&argument) {
                            if let Some(aliases) = references.aliases.get_mut(expression) {
                                // The argument reference is possibly aliased by this block parameter
                                aliases.insert(*parameter);
                            }
                        }
                    }
                }
            }
            TerminatorInstruction::Return { return_values } => {
                // Removing all `last_stores` for each returned reference is more important here
                // than setting them all to ReferenceValue::Unknown since no other block should
                // have a block with a Return terminator as a predecessor anyway.
                self.mark_all_unknown(return_values, function, references, last_stores);
            }
        }
    }
}

impl Block {
    /// If the given reference id points to a known value, return the value
    fn get_known_value(&self, address: ValueId) -> Option<ValueId> {
        if let Some(expression) = self.expressions.get(&address) {
            if let Some(aliases) = self.aliases.get(expression) {
                // We could allow multiple aliases if we check that the reference
                // value in each is equal.
                if aliases.len() == 1 {
                    let alias = aliases.first().expect("There should be exactly 1 alias");

                    if let Some(ReferenceValue::Known(value)) = self.references.get(alias) {
                        return Some(*value);
                    }
                }
            }
        }
        None
    }

    /// If the given address is known, set its value to `ReferenceValue::Known(value)`.
    fn set_known_value(
        &mut self,
        address: ValueId,
        value: ValueId,
        last_stores: &mut BTreeMap<ValueId, InstructionId>,
    ) {
        self.set_value(address, ReferenceValue::Known(value), last_stores);
    }

    fn set_unknown(
        &mut self,
        address: ValueId,
        last_stores: &mut BTreeMap<ValueId, InstructionId>,
    ) {
        self.set_value(address, ReferenceValue::Unknown, last_stores);
    }

    fn set_value(
        &mut self,
        address: ValueId,
        value: ReferenceValue,
        last_stores: &mut BTreeMap<ValueId, InstructionId>,
    ) {
        if let Some(expression) = self.expressions.get(&address) {
            if let Some(aliases) = self.aliases.get(expression) {
                if aliases.is_empty() {
                    // uh-oh, we don't know at all what this reference refers to, could be anything.
                    // Now we have to invalidate every reference we know of
                    self.invalidate_all_references(last_stores);
                } else if aliases.len() == 1 {
                    let alias = aliases.first().expect("There should be exactly 1 alias");
                    self.references.insert(*alias, value);
                } else {
                    // More than one alias. We're not sure which it refers to so we have to
                    // conservatively invalidate all references it may refer to.
                    for alias in aliases.iter() {
                        if let Some(reference_value) = self.references.get_mut(alias) {
                            *reference_value = ReferenceValue::Unknown;
                        }
                    }
                }
            }
        }
    }

    fn invalidate_all_references(&mut self, last_stores: &mut BTreeMap<ValueId, InstructionId>) {
        for reference_value in self.references.values_mut() {
            *reference_value = ReferenceValue::Unknown;
        }

        last_stores.clear();
    }

    fn unify(mut self, other: &Self) -> Self {
        for (value_id, expression) in &other.expressions {
            if let Some(existing) = self.expressions.get(value_id) {
                assert_eq!(existing, expression, "Expected expressions for {value_id} to be equal");
            } else {
                self.expressions.insert(*value_id, expression.clone());
            }
        }

        for (expression, new_aliases) in &other.aliases {
            let expression = expression.clone();

            self.aliases
                .entry(expression)
                .and_modify(|aliases| {
                    for alias in new_aliases {
                        aliases.insert(*alias);
                    }
                })
                .or_insert_with(|| new_aliases.clone());
        }

        // Keep only the references present in both maps.
        let mut intersection = BTreeMap::new();
        for (value_id, reference) in &other.references {
            if let Some(existing) = self.references.get(value_id) {
                intersection.insert(*value_id, existing.unify(*reference));
            }
        }
        self.references = intersection;

        self
    }

    /// Remember that `result` is the result of dereferencing `address`. This is important to
    /// track aliasing when references are stored within other references.
    fn remember_dereference(&mut self, function: &Function, address: ValueId, result: ValueId) {
        if function.dfg.type_of_value(result) == Type::Reference {
            if let Some(known_address) = self.get_known_value(address) {
                self.expressions.insert(result, Expression::Other(known_address));
            } else {
                let expression = Expression::Dereference(Box::new(Expression::Other(address)));
                self.expressions.insert(result, expression);
                // No known aliases to insert for this expression... can we find an alias
                // even if we don't have a known address? If not we'll have to invalidate all
                // known references if this reference is ever stored to.
            }
        }
    }
}

impl ReferenceValue {
    fn unify(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            ReferenceValue::Unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use acvm::FieldElement;
    use im::vector;

    use crate::ssa::{
        ir::{
            basic_block::BasicBlockId,
            dfg::DataFlowGraph,
            function::RuntimeType,
            instruction::{BinaryOp, Instruction, Intrinsic, TerminatorInstruction},
            map::Id,
            types::Type,
        },
        ssa_builder::FunctionBuilder,
    };

    #[test]
    fn test_simple() {
        // fn func() {
        //   b0():
        //     v0 = allocate
        //     store [Field 1, Field 2] in v0
        //     v1 = load v0
        //     v2 = array_get v1, index 1
        //     return v2
        // }

        let func_id = Id::test_new(0);
        let mut builder = FunctionBuilder::new("func".into(), func_id, RuntimeType::Acir);
        let v0 = builder.insert_allocate();
        let one = builder.field_constant(FieldElement::one());
        let two = builder.field_constant(FieldElement::one());

        let element_type = Rc::new(vec![Type::field()]);
        let array_type = Type::Array(element_type, 2);
        let array = builder.array_constant(vector![one, two], array_type.clone());

        builder.insert_store(v0, array);
        let v1 = builder.insert_load(v0, array_type);
        let v2 = builder.insert_array_get(v1, one, Type::field());
        builder.terminate_with_return(vec![v2]);

        let ssa = builder.finish().mem2reg().fold_constants();

        let func = ssa.main();
        let block_id = func.entry_block();

        assert_eq!(count_loads(block_id, &func.dfg), 0);
        assert_eq!(count_stores(block_id, &func.dfg), 0);

        let ret_val_id = match func.dfg[block_id].terminator().unwrap() {
            TerminatorInstruction::Return { return_values } => return_values.first().unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(func.dfg[*ret_val_id], func.dfg[two]);
    }

    #[test]
    fn test_simple_with_call() {
        // fn func {
        //   b0():
        //     v0 = allocate
        //     store v0, Field 1
        //     v1 = load v0
        //     call f0(v0)
        //     return v1
        // }

        let func_id = Id::test_new(0);
        let mut builder = FunctionBuilder::new("func".into(), func_id, RuntimeType::Acir);
        let v0 = builder.insert_allocate();
        let one = builder.field_constant(FieldElement::one());
        builder.insert_store(v0, one);
        let v1 = builder.insert_load(v0, Type::field());
        let f0 = builder.import_intrinsic_id(Intrinsic::AssertConstant);
        builder.insert_call(f0, vec![v0], vec![]);
        builder.terminate_with_return(vec![v1]);

        let ssa = builder.finish().mem2reg();

        let func = ssa.main();
        let block_id = func.entry_block();

        assert_eq!(count_loads(block_id, &func.dfg), 0);
        assert_eq!(count_stores(block_id, &func.dfg), 1);

        let ret_val_id = match func.dfg[block_id].terminator().unwrap() {
            TerminatorInstruction::Return { return_values } => return_values.first().unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(func.dfg[*ret_val_id], func.dfg[one]);
    }

    #[test]
    fn test_simple_with_return() {
        // fn func {
        //   b0():
        //     v0 = allocate
        //     store v0, Field 1
        //     return v0
        // }

        let func_id = Id::test_new(0);
        let mut builder = FunctionBuilder::new("func".into(), func_id, RuntimeType::Acir);
        let v0 = builder.insert_allocate();
        let const_one = builder.field_constant(FieldElement::one());
        builder.insert_store(v0, const_one);
        builder.terminate_with_return(vec![v0]);

        let ssa = builder.finish().mem2reg();

        let func = ssa.main();
        let block_id = func.entry_block();

        // Store is needed by the return value, and can't be removed
        assert_eq!(count_stores(block_id, &func.dfg), 1);

        let ret_val_id = match func.dfg[block_id].terminator().unwrap() {
            TerminatorInstruction::Return { return_values } => return_values.first().unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(func.dfg[*ret_val_id], func.dfg[v0]);
    }

    fn count_stores(block: BasicBlockId, dfg: &DataFlowGraph) -> usize {
        dfg[block]
            .instructions()
            .iter()
            .filter(|instruction_id| matches!(dfg[**instruction_id], Instruction::Store { .. }))
            .count()
    }

    fn count_loads(block: BasicBlockId, dfg: &DataFlowGraph) -> usize {
        dfg[block]
            .instructions()
            .iter()
            .filter(|instruction_id| matches!(dfg[**instruction_id], Instruction::Load { .. }))
            .count()
    }

    // Test that loads across multiple blocks are removed
    #[test]
    fn multiple_blocks() {
        // fn main {
        //   b0():
        //     v0 = allocate
        //     store Field 5 in v0
        //     v1 = load v0
        //     jmp b1(v1):
        //   b1(v2: Field):
        //     v3 = load v0
        //     store Field 6 in v0
        //     v4 = load v0
        //     return v2, v3, v4
        // }
        let main_id = Id::test_new(0);
        let mut builder = FunctionBuilder::new("main".into(), main_id, RuntimeType::Acir);

        let v0 = builder.insert_allocate();

        let five = builder.field_constant(5u128);
        builder.insert_store(v0, five);

        let v1 = builder.insert_load(v0, Type::field());
        let b1 = builder.insert_block();
        builder.terminate_with_jmp(b1, vec![v1]);

        builder.switch_to_block(b1);
        let v2 = builder.add_block_parameter(b1, Type::field());
        let v3 = builder.insert_load(v0, Type::field());

        let six = builder.field_constant(6u128);
        builder.insert_store(v0, six);
        let v4 = builder.insert_load(v0, Type::field());

        builder.terminate_with_return(vec![v2, v3, v4]);

        let ssa = builder.finish();
        assert_eq!(ssa.main().reachable_blocks().len(), 2);

        // Expected result:
        // acir fn main f0 {
        //   b0():
        //     v0 = allocate
        //     store Field 5 at v0
        //     jmp b1(Field 5)
        //   b1(v3: Field):
        //     store Field 6 at v0
        //     return v3, Field 5, Field 6 // Optimized to constants 5 and 6
        // }
        let ssa = ssa.mem2reg();

        let main = ssa.main();
        assert_eq!(main.reachable_blocks().len(), 2);

        // The loads should be removed
        assert_eq!(count_loads(main.entry_block(), &main.dfg), 0);
        assert_eq!(count_loads(b1, &main.dfg), 0);

        // Neither store is removed since they are each the last in the block and there are multiple blocks
        assert_eq!(count_stores(main.entry_block(), &main.dfg), 1);
        assert_eq!(count_stores(b1, &main.dfg), 1);

        // The jmp to b1 should also be a constant 5 now
        match main.dfg[main.entry_block()].terminator() {
            Some(TerminatorInstruction::Jmp { arguments, .. }) => {
                assert_eq!(arguments.len(), 1);
                let argument =
                    main.dfg.get_numeric_constant(arguments[0]).expect("Expected constant value");
                assert_eq!(argument.to_u128(), 5);
            }
            _ => unreachable!(),
        };
    }

    // Test that a load in a predecessor block has been removed if the value
    // is later stored in a successor block
    #[test]
    fn load_aliases_in_predecessor_block() {
        // fn main {
        //     b0():
        //       v0 = allocate
        //       store Field 0 at v0
        //       v2 = allocate
        //       store v0 at v2
        //       v3 = load v2
        //       v4 = load v2
        //       jmp b1()
        //     b1():
        //       store Field 1 at v3
        //       store Field 2 at v4
        //       v8 = load v3
        //       v9 = eq v8, Field 2
        //       return
        // }
        let main_id = Id::test_new(0);
        let mut builder = FunctionBuilder::new("main".into(), main_id, RuntimeType::Acir);

        let v0 = builder.insert_allocate();

        let zero = builder.field_constant(0u128);
        builder.insert_store(v0, zero);

        let v2 = builder.insert_allocate();
        builder.insert_store(v2, v0);

        let v3 = builder.insert_load(v2, Type::field());
        let v4 = builder.insert_load(v2, Type::field());
        let b1 = builder.insert_block();
        builder.terminate_with_jmp(b1, vec![]);

        builder.switch_to_block(b1);

        let one = builder.field_constant(1u128);
        builder.insert_store(v3, one);

        let two = builder.field_constant(2u128);
        builder.insert_store(v4, two);

        let v8 = builder.insert_load(v3, Type::field());
        let _ = builder.insert_binary(v8, BinaryOp::Eq, two);

        builder.terminate_with_return(vec![]);

        let ssa = builder.finish();
        assert_eq!(ssa.main().reachable_blocks().len(), 2);

        // Expected result:
        // acir fn main f0 {
        //   b0():
        //     v0 = allocate
        //     store Field 0 at v0
        //     v2 = allocate
        //     store v0 at v2
        //     jmp b1()
        //   b1():
        //     store Field 2 at v0
        //     v8 = eq Field 1, Field 2
        //     return
        // }
        let ssa = ssa.mem2reg();

        let main = ssa.main();
        assert_eq!(main.reachable_blocks().len(), 2);

        // All loads should be removed
        assert_eq!(count_loads(main.entry_block(), &main.dfg), 0);
        assert_eq!(count_loads(b1, &main.dfg), 0);

        // Only the first store in b1 is removed since there is another store to the same reference
        // in the same block, and the store is not needed before the later store.
        assert_eq!(count_stores(main.entry_block(), &main.dfg), 2);
        assert_eq!(count_stores(b1, &main.dfg), 1);

        let b1_instructions = main.dfg[b1].instructions();

        // The last instruction in b1 should be a binary operation
        match &main.dfg[*b1_instructions.last().unwrap()] {
            Instruction::Binary(binary) => {
                let lhs =
                    main.dfg.get_numeric_constant(binary.lhs).expect("Expected constant value");
                let rhs =
                    main.dfg.get_numeric_constant(binary.rhs).expect("Expected constant value");

                assert_eq!(lhs, rhs);
                assert_eq!(lhs, FieldElement::from(2u128));
            }
            _ => unreachable!(),
        }
    }
}
