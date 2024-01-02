use std::collections::{BTreeMap, BTreeSet};

use crate::{
    compiler::optimizers::GeneralOptimizer,
    pwg::{
        arithmetic::ExpressionSolver, directives::solve_directives, BrilligSolver,
        BrilligSolverStatus, OpcodeResolutionError,
    },
};
use acir::{
    circuit::{
        brillig::{Brillig, BrilligInputs},
        directives::{Directive, QuotientDirective},
        opcodes::BlackBoxFuncCall,
        Circuit, Opcode,
    },
    native_types::{Expression, Witness, WitnessMap},
    BlackBoxFunc, FieldElement,
};
use acvm_blackbox_solver::{BlackBoxFunctionSolver, BlackBoxResolutionError};

/// `RangeOptimizer` will remove redundant range constraints.
///
/// # Example
///
/// Suppose we had the following pseudo-code:
///
/// ```noir
/// let z1 = x as u16;
/// let z2 = x as u32;
/// ```
/// It is clear that if `x` fits inside of a 16-bit integer,
/// it must also fit inside of a 32-bit integer.
///
/// The generated ACIR may produce two range opcodes however;
/// - One for the 16 bit range constraint of `x`
/// - One for the 32-bit range constraint of `x`
///
/// This optimization pass will keep the 16-bit range constraint
/// and remove the 32-bit range constraint opcode.
pub(crate) struct ConstantBackpropOptimizer {
    circuit: Circuit,
}

impl ConstantBackpropOptimizer {
    /// Creates a new `ConstantBackpropOptimizer`
    pub(crate) fn new(circuit: Circuit) -> Self {
        Self { circuit }
    }

    fn gather_known_witnesses(&self) -> WitnessMap {
        // We do not want to affect the circuit's interface so avoid optimizing away these witnesses.
        let mut required_witnesses: BTreeSet<Witness> = self
            .circuit
            .private_parameters
            .union(&self.circuit.public_parameters.0)
            .chain(&self.circuit.return_values.0)
            .copied()
            .collect();

        for opcode in &self.circuit.opcodes {
            match &opcode {
                Opcode::BlackBoxFuncCall(func_call) => {
                    required_witnesses.extend(
                        func_call.get_inputs_vec().into_iter().map(|func_input| func_input.witness),
                    );
                }

                Opcode::MemoryInit { init, .. } => {
                    required_witnesses.extend(init);
                }

                Opcode::MemoryOp { op, .. } => {
                    required_witnesses.insert(op.index.to_witness().unwrap());
                    required_witnesses.insert(op.value.to_witness().unwrap());
                }

                _ => (),
            };
        }

        let mut known_witnesses = WitnessMap::new();
        for opcode in self.circuit.opcodes.iter().rev() {
            if let Opcode::AssertZero(expr) = opcode {
                match ExpressionSolver::solve(&mut known_witnesses, expr) {
                    Ok(()) | Err(OpcodeResolutionError::OpcodeNotSolvable(_)) => {
                        // println!("failed");
                    }
                    Err(_) => todo!(),
                }
            }
        }

        // We want to retain any references to required witnesses so we "forget" these assignments.
        let known_witnesses: BTreeMap<_, _> = known_witnesses
            .into_iter()
            .filter(|(witness, _)| !required_witnesses.contains(witness))
            .collect();

        known_witnesses.into()
    }

    /// Returns a `Circuit` where each Witness is only range constrained
    /// once to the lowest number `bit size` possible.
    #[tracing::instrument(level = "trace", skip_all)]
    pub(crate) fn backpropagate_constants(
        circuit: Circuit,
        order_list: Vec<usize>,
    ) -> (Circuit, Vec<usize>) {
        let old_circuit_size = circuit.opcodes.len();

        let optimizer = Self::new(circuit);
        let (circuit, order_list) = optimizer.backpropagate_constants_iteration(order_list);

        let new_circuit_size = circuit.opcodes.len();
        if new_circuit_size < old_circuit_size {
            Self::backpropagate_constants(circuit, order_list)
        } else {
            (circuit, order_list)
        }
    }

    /// Returns a `Circuit` where each Witness is only range constrained
    /// once to the lowest number `bit size` possible.
    pub(crate) fn backpropagate_constants_iteration(
        self,
        order_list: Vec<usize>,
    ) -> (Circuit, Vec<usize>) {
        let mut known_witnesses = self.gather_known_witnesses();

        let Circuit {
            current_witness_index,
            opcodes,
            private_parameters,
            public_parameters,
            return_values,
            assert_messages,
        } = self.circuit;

        fn remap_expression(known_witnesses: &WitnessMap, expression: Expression) -> Expression {
            GeneralOptimizer::optimize(ExpressionSolver::evaluate(&expression, known_witnesses))
        }

        let mut new_order_list = Vec::with_capacity(order_list.len());
        let mut new_opcodes = Vec::with_capacity(opcodes.len());
        for (idx, opcode) in opcodes.into_iter().enumerate() {
            let new_opcode = match opcode {
                Opcode::AssertZero(expression) => {
                    let new_expr = remap_expression(&known_witnesses, expression);
                    if new_expr.is_zero() {
                        continue;
                    }

                    // Attempt to solve the opcode to see if we can determine the value of any witnesses in the expression.
                    // We only do this _after_ we apply any simplifications to create the new opcode as we want to
                    // keep the constraint on the witness which we can solving for here.
                    let _ = ExpressionSolver::solve(&mut known_witnesses, &new_expr);

                    Opcode::AssertZero(new_expr)
                }
                Opcode::Brillig(brillig) => {
                    let remapped_inputs = brillig
                        .inputs
                        .into_iter()
                        .map(|input| match input {
                            BrilligInputs::Single(expr) => {
                                BrilligInputs::Single(remap_expression(&known_witnesses, expr))
                            }
                            BrilligInputs::Array(expr_array) => {
                                let new_input: Vec<_> = expr_array
                                    .into_iter()
                                    .map(|expr| remap_expression(&known_witnesses, expr))
                                    .collect();

                                BrilligInputs::Array(new_input)
                            }
                        })
                        .collect();

                    let remapped_predicate = brillig
                        .predicate
                        .map(|predicate| remap_expression(&known_witnesses, predicate));

                    let new_brillig = Brillig {
                        inputs: remapped_inputs,
                        predicate: remapped_predicate,
                        ..brillig
                    };

                    struct NullBbSolver;

                    impl BlackBoxFunctionSolver for NullBbSolver {
                        fn schnorr_verify(
                            &self,
                            _public_key_x: &FieldElement,
                            _public_key_y: &FieldElement,
                            _signature: &[u8],
                            _message: &[u8],
                        ) -> Result<bool, BlackBoxResolutionError> {
                            Err(BlackBoxResolutionError::Failed(
                                BlackBoxFunc::SchnorrVerify,
                                "".to_owned(),
                            ))
                        }

                        fn pedersen_commitment(
                            &self,
                            _inputs: &[FieldElement],
                            _domain_separator: u32,
                        ) -> Result<(FieldElement, FieldElement), BlackBoxResolutionError>
                        {
                            Err(BlackBoxResolutionError::Failed(
                                BlackBoxFunc::PedersenCommitment,
                                "".to_owned(),
                            ))
                        }
                        fn pedersen_hash(
                            &self,
                            _inputs: &[FieldElement],
                            _domain_separator: u32,
                        ) -> Result<FieldElement, BlackBoxResolutionError> {
                            Err(BlackBoxResolutionError::Failed(
                                BlackBoxFunc::PedersenHash,
                                "".to_owned(),
                            ))
                        }

                        fn fixed_base_scalar_mul(
                            &self,
                            _low: &FieldElement,
                            _high: &FieldElement,
                        ) -> Result<(FieldElement, FieldElement), BlackBoxResolutionError>
                        {
                            Err(BlackBoxResolutionError::Failed(
                                BlackBoxFunc::FixedBaseScalarMul,
                                "".to_owned(),
                            ))
                        }
                    }

                    let backend = NullBbSolver;

                    if let Ok(mut solver) =
                        BrilligSolver::new(&known_witnesses, &new_brillig, &backend, idx)
                    {
                        match solver.solve() {
                            Ok(BrilligSolverStatus::Finished) => {
                                // Write execution outputs
                                match solver.finalize(&mut known_witnesses, &new_brillig) {
                                    Ok(()) => continue,
                                    _ => Opcode::Brillig(new_brillig),
                                }
                            }
                            _ => Opcode::Brillig(new_brillig),
                        }
                    } else {
                        Opcode::Brillig(new_brillig)
                    }
                }

                Opcode::Directive(directive) => {
                    let directive = match directive {
                        Directive::Quotient(QuotientDirective { a, b, q, r, predicate }) => {
                            Directive::Quotient(QuotientDirective {
                                a: remap_expression(&known_witnesses, a),
                                b: remap_expression(&known_witnesses, b),
                                q,
                                r,
                                predicate: predicate
                                    .map(|predicate| remap_expression(&known_witnesses, predicate)),
                            })
                        }
                        Directive::ToLeRadix { a, b, radix } => Directive::ToLeRadix {
                            a: remap_expression(&known_witnesses, a),
                            b,
                            radix,
                        },
                        Directive::PermutationSort { inputs, tuple, bits, sort_by } => {
                            Directive::PermutationSort {
                                inputs: inputs
                                    .into_iter()
                                    .map(|expr_vec| {
                                        expr_vec
                                            .into_iter()
                                            .map(|expr| remap_expression(&known_witnesses, expr))
                                            .collect()
                                    })
                                    .collect(),
                                tuple,
                                bits,
                                sort_by,
                            }
                        }
                    };

                    match solve_directives(&mut known_witnesses, &directive) {
                        Ok(()) => continue,
                        Err(_) => Opcode::Directive(directive),
                    }
                }

                Opcode::BlackBoxFuncCall(BlackBoxFuncCall::RANGE { input }) => {
                    match known_witnesses.get(&input.witness) {
                        Some(known_value) if known_value.num_bits() <= input.num_bits => {
                            continue;
                        }
                        _ => opcode,
                    }
                }

                Opcode::BlackBoxFuncCall(_)
                | Opcode::MemoryOp { .. }
                | Opcode::MemoryInit { .. } => opcode,
            };

            new_opcodes.push(new_opcode);
            new_order_list.push(order_list[idx]);
        }

        (
            Circuit {
                current_witness_index,
                opcodes: new_opcodes,
                private_parameters,
                public_parameters,
                return_values,
                assert_messages,
            },
            new_order_list,
        )
    }
}
