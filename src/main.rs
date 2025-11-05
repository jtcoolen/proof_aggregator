// Keep all your existing `use` and type aliases from the original snippet.
// This file shows a *fixed* version of Agg2HCircuit with configurable tree height
// and a driver that builds a binary aggregation tree. It addresses the compile
// errors you reported (wrong VK/PK types, missing SRS type, bit/cell mismatches)
// and removes use of committed instances (mirroring the IVC example).

use halo2curves::{ff::Field, group::Group};
use midnight_circuits::hash::poseidon::PoseidonState;
use midnight_circuits::types::AssignedForeignPoint;
use midnight_circuits::verifier::Msm;
use midnight_circuits::{
    ecc::{
        curves::CircuitCurve,
        foreign::{ForeignEccChip, ForeignEccConfig, nb_foreign_ecc_chip_columns},
    },
    field::{
        NativeChip, NativeConfig, NativeGadget,
        decomposition::{
            chip::{P2RDecompositionChip, P2RDecompositionConfig},
            pow2range::Pow2RangeChip,
        },
        foreign::FieldChip,
        native::NB_ARITH_COLS,
    },
    hash::poseidon::{
        NB_POSEIDON_ADVICE_COLS, NB_POSEIDON_FIXED_COLS, PoseidonChip, PoseidonConfig,
    },
    instructions::*,
    types::{AssignedNative, ComposableChip, Instantiable},
    verifier::{
        self, Accumulator, AssignedAccumulator, AssignedVk, BlstrsEmulation, SelfEmulation,
        VerifierGadget,
    },
};
use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG; // <-- correct path for private ParamsKZG
use midnight_proofs::utils::SerdeFormat;
use midnight_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error, create_proof, keygen_pk, keygen_vk_with_k, prepare},
    poly::{EvaluationDomain, kzg::KZGCommitmentScheme},
    transcript::{CircuitTranscript, Transcript},
};
use rand::rngs::OsRng;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;
use std::time::Instant;

// Reuse your type aliases
type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;

type E = <S as SelfEmulation>::Engine;
type CBase = <C as CircuitCurve>::Base;

type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

const K: u32 = 20;

/// Use filecoin's SRS (over BLS12-381)
fn filecoin_srs(k: u32) -> ParamsKZG<Bls12> {
    assert!(k <= 20, "We don't have an SRS for circuits of size {k}");

    let srs_dir = env::var("SRS_DIR").unwrap_or("./examples/assets".into());

    let srs_path = format!("{srs_dir}/bls_filecoin_2p{k:?}");
    let mut fetching_path = srs_path.clone();

    if !Path::new(fetching_path.as_str()).exists() {
        fetching_path = format!("{srs_dir}/bls_filecoin_2p20")
    }

    let params_fs = File::open(Path::new(&fetching_path)).unwrap_or_else(|_| {
        panic!("\nIt seems you have not downloaded and/or parsed the SRS from filecoin.")
    });

    let mut params: ParamsKZG<Bls12> = ParamsKZG::read_custom::<_>(
        &mut BufReader::new(params_fs),
        SerdeFormat::RawBytesUnchecked,
    )
    .expect("Failed to read params");

    if fetching_path != srs_path {
        params.downsize(k);

        let mut buf = Vec::new();

        params
            .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
            .expect("Failed to write params");
        let mut file = File::create(srs_path).expect("Failed to create file");

        file.write_all(&buf[..])
            .expect("Failed to write params to file");
    }

    params
}

#[derive(Clone, Debug)]
pub struct IvcCircuit {
    self_vk: (EvaluationDomain<F>, ConstraintSystem<F>, Value<F>), // (domain, cs, vk_repr)
    // We use a simple application function that increases a counter.
    left_state: Value<F>,
    right_state: Value<F>,
    left_proof: Value<Vec<u8>>,
    right_proof: Value<Vec<u8>>,
    left_acc: Value<Accumulator<S>>,
    right_acc: Value<Accumulator<S>>,
}

fn configure_ivc_circuit(
    meta: &mut ConstraintSystem<F>,
) -> (
    NativeConfig,
    P2RDecompositionConfig,
    ForeignEccConfig<C>,
    PoseidonConfig<F>,
) {
    let nb_advice_cols = nb_foreign_ecc_chip_columns::<F, C, C, NG>();
    let nb_fixed_cols = NB_ARITH_COLS + 4;

    let advice_columns: Vec<_> = (0..nb_advice_cols).map(|_| meta.advice_column()).collect();
    let fixed_columns: Vec<_> = (0..nb_fixed_cols).map(|_| meta.fixed_column()).collect();
    // We still configure two instance columns for compatibility with the chips,
    // but we will NOT use the committed one at proving time (it will be empty).
    let committed_instance_column = meta.instance_column(); // column 0 (unused/empty)
    let instance_column = meta.instance_column(); // column 1 (plain)

    let native_config = NativeChip::configure(
        meta,
        &(
            advice_columns[..NB_ARITH_COLS].try_into().unwrap(),
            fixed_columns[..NB_ARITH_COLS + 4].try_into().unwrap(),
            [committed_instance_column, instance_column],
        ),
    );
    let core_decomp_config = {
        let pow2_config = Pow2RangeChip::configure(meta, &advice_columns[1..NB_ARITH_COLS]);
        P2RDecompositionChip::configure(meta, &(native_config.clone(), pow2_config))
    };

    let base_config = FieldChip::<F, CBase, C, NG>::configure(meta, &advice_columns);
    let curve_config =
        ForeignEccChip::<F, C, C, NG, NG>::configure(meta, &base_config, &advice_columns);

    let poseidon_config = PoseidonChip::configure(
        meta,
        &(
            advice_columns[..NB_POSEIDON_ADVICE_COLS]
                .try_into()
                .unwrap(),
            fixed_columns[..NB_POSEIDON_FIXED_COLS].try_into().unwrap(),
        ),
    );

    (
        native_config,
        core_decomp_config,
        curve_config,
        poseidon_config,
    )
}

impl Circuit<F> for IvcCircuit {
    type Config = (
        NativeConfig,
        P2RDecompositionConfig,
        ForeignEccConfig<C>,
        PoseidonConfig<F>,
    );
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        unreachable!()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        configure_ivc_circuit(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let native_chip = <NativeChip<F> as ComposableChip<F>>::new(&config.0, &());
        let core_decomp_chip = P2RDecompositionChip::new(&config.1, &(K as usize - 1));
        let scalar_chip = NativeGadget::new(core_decomp_chip.clone(), native_chip.clone());
        let curve_chip = { ForeignEccChip::new(&config.2, &scalar_chip, &scalar_chip) };
        let poseidon_chip = PoseidonChip::new(&config.3, &native_chip);

        let verifier_chip = VerifierGadget::new(&curve_chip, &scalar_chip, &poseidon_chip);

        // === VK is constrained as PUBLIC INPUT in the *plain* instance column ===
        let self_vk_name = "self_vk";
        let (self_domain, self_cs, self_vk_value) = &self.self_vk;
        let assigned_self_vk: AssignedVk<S> = verifier_chip.assign_vk_as_public_input(
            &mut layouter,
            self_vk_name,
            self_domain,
            self_cs,
            *self_vk_value,
        )?;

        // Witness left and right states and compute the new state.
        // Change: "genesis" (both child states == 0) yields next_state = 1.
        // Otherwise, next_state = left_state + right_state.
        // Constrain the new state as a PUBLIC INPUT (PLAIN instance column).
        let left_state = scalar_chip.assign(&mut layouter, self.left_state)?;
        let right_state = scalar_chip.assign(&mut layouter, self.right_state)?;

        // Bits indicating whether each child is zero.
        let is_left_zero = scalar_chip.is_zero(&mut layouter, &left_state)?;
        let is_right_zero = scalar_chip.is_zero(&mut layouter, &right_state)?;
        // "Genesis" bit: both zero -> 1, else 0 (use multiplication for AND on {0,1} bits)
        let is_genesis = scalar_chip.and(
            &mut layouter,
            &[is_left_zero.clone(), is_right_zero.clone()],
        )?;
        // next_state = left + right + is_genesis
        let tmp_sum = scalar_chip.add(&mut layouter, &left_state, &right_state)?;
        let next_state = scalar_chip.add(&mut layouter, &tmp_sum, &is_genesis.into())?;
        scalar_chip.constrain_as_public_input(&mut layouter, &next_state)?;

        // Fixed "committed instance" base used by the verifier (identity point).
        // We keep this for compatibility with the verifier gadget,
        // but no committed scalars are provided at proving time.
        let id_point: AssignedForeignPoint<F, C, _> =
            curve_chip.assign_fixed(&mut layouter, C::identity())?;

        let mut fixed_base_names = vec![String::from("com_instance")];
        fixed_base_names.extend(verifier::fixed_base_names::<S>(
            self_vk_name,
            self_cs.num_fixed_columns() + self_cs.num_selectors(),
            self_cs.permutation().columns.len(),
        ));

        // ----- Left child -----
        let left_acc = AssignedAccumulator::assign(
            &mut layouter,
            &curve_chip,
            &scalar_chip,
            1,
            1,
            &[],
            &fixed_base_names,
            self.left_acc.clone(),
        )?;

        let assigned_left_pi = [
            verifier_chip.as_public_input(&mut layouter, &assigned_self_vk)?,
            vec![left_state.clone()],
            verifier_chip.as_public_input(&mut layouter, &left_acc)?,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        let mut left_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_self_vk,
            &[("com_instance", id_point.clone())],
            &[&assigned_left_pi],
            self.left_proof.clone(),
        )?;

        // Genesis gating for left child (now uses the precomputed zero-bit)
        let is_not_left_genesis = scalar_chip.not(&mut layouter, &is_left_zero)?;
        AssignedAccumulator::scale_by_bit(
            &mut layouter,
            &scalar_chip,
            &is_not_left_genesis,
            &mut left_proof_acc,
        )?;
        left_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        // ----- Right child -----
        let right_acc = AssignedAccumulator::assign(
            &mut layouter,
            &curve_chip,
            &scalar_chip,
            1,
            1,
            &[],
            &fixed_base_names,
            self.right_acc.clone(),
        )?;

        let assigned_right_pi = [
            verifier_chip.as_public_input(&mut layouter, &assigned_self_vk)?,
            vec![right_state.clone()],
            verifier_chip.as_public_input(&mut layouter, &right_acc)?,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        let mut right_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_self_vk,
            &[("com_instance", id_point)],
            &[&assigned_right_pi],
            self.right_proof.clone(),
        )?;

        // Genesis gating for right child (uses the precomputed zero-bit)
        let is_not_right_genesis = scalar_chip.not(&mut layouter, &is_right_zero)?;
        AssignedAccumulator::scale_by_bit(
            &mut layouter,
            &scalar_chip,
            &is_not_right_genesis,
            &mut right_proof_acc,
        )?;
        right_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        // Accumulate: left_proof_acc, left_acc, right_proof_acc, right_acc
        let mut next_acc = AssignedAccumulator::<S>::accumulate(
            &mut layouter,
            &verifier_chip,
            &scalar_chip,
            &poseidon_chip,
            &[left_proof_acc, left_acc, right_proof_acc, right_acc],
        )?;

        // Collapse and constrain the accumulator as PUBLIC INPUT (PLAIN column).
        next_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;
        verifier_chip.constrain_as_public_input(&mut layouter, &next_acc)?;

        core_decomp_chip.load(&mut layouter)
    }
}

#[derive(Clone, Debug)]
struct TreeNode {
    state: F,
    proof: Vec<u8>,
    // Accumulator derived from verifying THIS node's proof (used by its parent as `*_proof_acc`)
    proof_acc: Accumulator<S>,
    // Accumulator that THIS node exposed as a public input (used by its parent as `*_acc`)
    pi_acc: Accumulator<S>,
}

fn zero_acc() -> Accumulator<S> {
    Accumulator::<S>::new(
        Msm::new(&[], &[], &BTreeMap::new()),
        Msm::new(&[], &[], &BTreeMap::new()),
    )
}

fn main() {
    let self_k = K;

    let mut self_cs = ConstraintSystem::default();
    configure_ivc_circuit(&mut self_cs);
    let self_domain = EvaluationDomain::new(self_cs.degree() as u32, self_k);

    let default_ivc_circuit = IvcCircuit {
        self_vk: (self_domain.clone(), self_cs.clone(), Value::unknown()),
        left_state: Value::unknown(),
        right_state: Value::unknown(),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_acc: Value::unknown(),
        right_acc: Value::unknown(),
    };

    let srs = filecoin_srs(self_k);

    let start = Instant::now();
    let vk = keygen_vk_with_k(&srs, &default_ivc_circuit, self_k).unwrap();
    let pk = keygen_pk(vk.clone(), &default_ivc_circuit).unwrap();
    println!("Computed vk and pk in {:?} s", start.elapsed());

    let mut fixed_bases = BTreeMap::new();
    fixed_bases.insert(String::from("com_instance"), C::identity());
    fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "self_vk", &vk,
    ));
    let fixed_base_names = fixed_bases.keys().cloned().collect::<Vec<_>>();

    // This trivial accumulator must have a single base and scalar of F::ONE, and
    // the base has to be the default point of C.
    let trivial_acc = Accumulator::<S>::new(
        Msm::new(&[C::default()], &[F::ONE], &BTreeMap::new()),
        Msm::new(
            &[C::default()],
            &[F::ONE],
            &fixed_base_names
                .iter()
                .map(|name| (name.clone(), F::ZERO))
                .collect(),
        ),
    );

    // Create leaf nodes (genesis nodes). With left=0 and right=0,
    // the circuit now forces next_state = 1 (genesis bias).
    let num_leaves = 4; // Create a tree with 4 leaves (can be adjusted)
    let mut current_level: Vec<TreeNode> = vec![];

    println!("Creating {} leaf nodes...", num_leaves);
    for i in 0..num_leaves {
        // For genesis leaves both child states are zero so both child proofs are skipped in-circuit.
        // But the leaf's own state is 1.
        let state = F::ONE;

        let circuit = IvcCircuit {
            self_vk: (
                self_domain.clone(),
                self_cs.clone(),
                Value::known(vk.transcript_repr()),
            ),
            left_state: Value::known(F::ZERO),
            right_state: Value::known(F::ZERO),
            left_proof: Value::known(vec![]),
            right_proof: Value::known(vec![]),
            left_acc: Value::known(trivial_acc.clone()), // child's PI acc (leaf)
            right_acc: Value::known(trivial_acc.clone()), // child's PI acc (leaf)
        };

        // === Single plain-column public inputs: [ VK || state || accumulator ] ===
        // For a leaf, the PI accumulator is the trivial one.
        let mut public_inputs = AssignedVk::<S>::as_public_input(&vk); // VK (now plain)
        public_inputs.extend(AssignedNative::<F>::as_public_input(&state));
        public_inputs.extend(AssignedAccumulator::as_public_input(&trivial_acc));

        let start = Instant::now();
        let proof = {
            let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
            create_proof::<
                F,
                KZGCommitmentScheme<E>,
                CircuitTranscript<PoseidonState<F>>,
                IvcCircuit,
            >(
                &srs,
                &pk,
                &[circuit.clone()],
                1,
                // Pass [ empty committed , plain public inputs ]
                &[&[&[], &public_inputs]],
                OsRng,
                &mut transcript,
            )
            .unwrap_or_else(|_| panic!("Problem creating leaf {i} proof"));
            transcript.finalize()
        };
        println!("Leaf {i} proof created in {:?}", start.elapsed());

        // proof_acc for THIS node (what the parent will use as `left/right_proof_acc`)
        let proof_acc = verify_and_extract_acc(&srs, &vk, &fixed_bases, &proof, &public_inputs);

        // pi_acc for THIS node (what the parent will witness as `left/right_acc`)
        let pi_acc = trivial_acc.clone();

        current_level.push(TreeNode {
            state,
            proof,
            proof_acc,
            pi_acc,
        });
    }

    // Build the tree level by level
    let mut level = 0;
    while current_level.len() > 1 {
        level += 1;
        println!(
            "\nBuilding level {} with {} nodes...",
            level,
            current_level.len() / 2
        );

        let mut next_level = vec![];

        for i in (0..current_level.len()).step_by(2) {
            let left = &current_level[i];
            let right = &current_level[i + 1];

            // For internal nodes, next_state = left.state + right.state (no +1).
            let state = left.state + right.state;

            // Parent circuit must witness each child's PI accumulator (not their proof_acc).
            let circuit = IvcCircuit {
                self_vk: (
                    self_domain.clone(),
                    self_cs.clone(),
                    Value::known(vk.transcript_repr()),
                ),
                left_state: Value::known(left.state),
                right_state: Value::known(right.state),
                left_proof: Value::known(left.proof.clone()),
                right_proof: Value::known(right.proof.clone()),
                left_acc: Value::known(left.pi_acc.clone()),
                right_acc: Value::known(right.pi_acc.clone()),
            };

            // Apply the SAME genesis gating as in-circuit:
            let left_proof_component = if left.state == F::ZERO {
                zero_acc()
            } else {
                left.proof_acc.clone()
            };
            let right_proof_component = if right.state == F::ZERO {
                zero_acc()
            } else {
                right.proof_acc.clone()
            };

            // Compute the parent PI accumulator EXACTLY as in-circuit order:
            // accumulate([left_proof_acc, left_pi_acc, right_proof_acc, right_pi_acc])
            let mut accumulated_pi = Accumulator::accumulate(&[
                left_proof_component,
                left.pi_acc.clone(),
                right_proof_component,
                right.pi_acc.clone(),
            ]);
            accumulated_pi.collapse();

            // === Single plain-column public inputs for internal node ===
            let mut public_inputs = AssignedVk::<S>::as_public_input(&vk);
            public_inputs.extend(AssignedNative::<F>::as_public_input(&state));
            public_inputs.extend(AssignedAccumulator::as_public_input(&accumulated_pi));

            let start = Instant::now();
            let proof = {
                let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<E>,
                    CircuitTranscript<PoseidonState<F>>,
                    IvcCircuit,
                >(
                    &srs,
                    &pk,
                    &[circuit.clone()],
                    1,
                    // [ empty committed , plain public inputs ]
                    &[&[&[], &public_inputs]],
                    OsRng,
                    &mut transcript,
                )
                .unwrap_or_else(|_| panic!("Problem creating level {level} node {}", i / 2));
                transcript.finalize()
            };
            println!(
                "Level {} node {} proof created in {:?}",
                level,
                i / 2,
                start.elapsed()
            );

            // proof_acc for THIS node (what the *next* level will use as child proof_acc)
            let proof_acc = verify_and_extract_acc(&srs, &vk, &fixed_bases, &proof, &public_inputs);

            // pi_acc for THIS node (what the *next* level will witness as child acc)
            let pi_acc = accumulated_pi;

            println!("Asserted validity of state {:?}", state);

            next_level.push(TreeNode {
                state,
                proof,
                proof_acc,
                pi_acc,
            });
        }

        current_level = next_level;
    }

    let root = &current_level[0];
    println!("\n=== Tree Construction Complete ===");
    println!("Root state: {:?}", root.state);

    // With the current leaf construction (left=0, right=0, genesis bias +1),
    // each leaf state is 1. For a binary tree that sums child states at internal
    // nodes, the expected root state is the number of leaves.
    println!(
        "Expected root state (sum of {} leaves): {:?}",
        num_leaves,
        F::from(num_leaves as u64)
    );
}

// NOTE: This verifier expects ALL public inputs (VK || state || acc) in the PLAIN instance column.
// The "committed base" is still supplied (identity) for compatibility, but there are no committed scalars.
fn verify_and_extract_acc(
    srs: &ParamsKZG<Bls12>,
    vk: &midnight_proofs::plonk::VerifyingKey<F, KZGCommitmentScheme<E>>,
    fixed_bases: &BTreeMap<String, C>,
    proof: &[u8],
    plain_public_inputs: &[F],
) -> Accumulator<S> {
    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);

    // committed instance bases: keep a single identity base for compatibility
    let committed_bases: &[&[C]] = &[&[C::identity()]];
    // instances (plain columns): one plain column containing [vk || next_state || accumulator scalars]
    let instances: &[&[&[F]]] = &[&[plain_public_inputs]];

    let dual_msm = prepare::<F, KZGCommitmentScheme<E>, CircuitTranscript<PoseidonState<F>>>(
        vk,
        committed_bases,
        instances,
        &mut transcript,
    )
    .expect("Verification failed");

    assert!(dual_msm.clone().check(&srs.verifier_params()));

    let mut acc: Accumulator<S> = dual_msm.into();
    acc.extract_fixed_bases(fixed_bases);
    acc.collapse();

    assert!(
        acc.check(&srs.s_g2().into(), fixed_bases),
        "Accumulator verification failed"
    );

    acc
}
