use halo2curves::{ff::Field, group::Group};
use midnight_circuits::compact_std_lib::MidnightCircuit;
use midnight_circuits::hash::poseidon::PoseidonState;
use midnight_circuits::types::AssignedForeignPoint;
use midnight_circuits::{
    compact_std_lib::{self, Relation, ZkStdLib, ZkStdLibArch},
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
    verifier::{Accumulator, AssignedAccumulator, BlstrsEmulation, SelfEmulation, VerifierGadget},
};
use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use midnight_proofs::utils::SerdeFormat;
use midnight_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Circuit, ConstraintSystem, Error, VerifyingKey, create_proof, keygen_pk, keygen_vk_with_k,
    },
    poly::{EvaluationDomain, kzg::KZGCommitmentScheme},
    transcript::{CircuitTranscript, Transcript},
};
use rand::rngs::OsRng;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;
type CBase = <C as CircuitCurve>::Base;
type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

const K: u32 = 20;
const POSEIDON_K: u32 = 6;

fn filecoin_srs(k: u32) -> ParamsKZG<Bls12> {
    assert!(k <= 20, "We don't have an SRS for circuits of size {}", k);
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

#[derive(Clone, Default)]
pub struct PoseidonExample;

impl Relation for PoseidonExample {
    type Instance = F;
    type Witness = [F; 3];

    fn format_instance(instance: &Self::Instance) -> Result<Vec<F>, Error> {
        Ok(vec![*instance])
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<F>,
        _instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        let assigned_message = std_lib.assign_many(layouter, &witness.transpose_array())?;
        let output = std_lib.poseidon(layouter, &assigned_message)?;
        std_lib.constrain_as_public_input(layouter, &output)
    }

    fn used_chips(&self) -> ZkStdLibArch {
        ZkStdLibArch {
            jubjub: false,
            poseidon: true,
            sha256: false,
            sha512: false,
            secp256k1: false,
            bls12_381: false,
            base64: false,
            nr_pow2range_cols: 1,
            automaton: false,
        }
    }

    fn write_relation<W: std::io::Write>(&self, _writer: &mut W) -> std::io::Result<()> {
        Ok(())
    }

    fn read_relation<R: std::io::Read>(_reader: &mut R) -> std::io::Result<Self> {
        Ok(PoseidonExample)
    }
}

#[derive(Clone, Debug)]
pub struct AggCircuit {
    // The VK *of the circuit verified at this layer* (i.e. children).
    // We keep (domain, cs) for the verifier gadget and a fixed transcript repr for in-circuit VK PI hashing.
    child_vk: (EvaluationDomain<F>, ConstraintSystem<F>, F),
    child_vk_name: String,

    // The VK repr that the *children* embedded in their own PI hash (i.e. their vk_for_pi).
    // Unused when `is_leaf == true`.
    inner_vk_repr: F,

    // Enforced `prev_level` value for this circuit variant (0 for leaf agg, >=1 for internal levels).
    expected_prev_level: F,

    left_state: Value<F>,
    right_state: Value<F>,
    left_proof: Value<Vec<u8>>,
    right_proof: Value<Vec<u8>>,
    left_acc: Value<Accumulator<S>>,
    right_acc: Value<Accumulator<S>>,
    fixed_base_names: Vec<String>,
    prev_level: Value<F>,
    is_leaf: bool,
}

fn configure_agg_circuit(
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
    let committed_instance_column = meta.instance_column();
    let instance_column = meta.instance_column();

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

impl Circuit<F> for AggCircuit {
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
        configure_agg_circuit(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let native_chip = <NativeChip<F> as ComposableChip<F>>::new(&config.0, &());
        let core_decomp_chip = P2RDecompositionChip::new(&config.1, &(K as usize - 1));
        let scalar_chip = NativeGadget::new(core_decomp_chip.clone(), native_chip.clone());
        let curve_chip = ForeignEccChip::new(&config.2, &scalar_chip, &scalar_chip);
        let poseidon_chip = PoseidonChip::new(&config.3, &native_chip);
        let verifier_chip = VerifierGadget::new(&curve_chip, &scalar_chip, &poseidon_chip);

        // Enforce expected level and compute next level
        let prev_level = scalar_chip.assign(&mut layouter, self.prev_level)?;
        scalar_chip.assert_equal_to_fixed(&mut layouter, &prev_level, self.expected_prev_level)?;
        let next_level = scalar_chip.add_constant(&mut layouter, &prev_level, F::ONE)?;

        // Child VK transcript representation is a fixed constant in-circuit for this layer
        let child_vk_val: AssignedNative<F> =
            native_chip.assign_fixed(&mut layouter, self.child_vk.2)?;

        // Compute next state (Poseidon over children states)
        let left_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.left_state)?;
        let right_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.right_state)?;
        let next_state =
            poseidon_chip.hash(&mut layouter, &[left_state.clone(), right_state.clone()])?;

        // Assigned accumulators for children (these are *PI accumulators* provided as witnesses)
        let mut left_acc = AssignedAccumulator::assign(
            &mut layouter,
            &curve_chip,
            &scalar_chip,
            1,
            1,
            &[],
            &self.fixed_base_names,
            self.left_acc.clone(),
        )?;
        let mut right_acc = AssignedAccumulator::assign(
            &mut layouter,
            &curve_chip,
            &scalar_chip,
            1,
            1,
            &[],
            &self.fixed_base_names,
            self.right_acc.clone(),
        )?;

        // VK used by the partial verifier at this layer (always the "child_vk" of this layer)
        let assigned_vk = verifier_chip.assign_vk(
            self.child_vk_name.as_str(),
            &self.child_vk.0,
            &self.child_vk.1,
            child_vk_val.clone(),
        )?;

        // Compute the child public inputs expected by the verifier gadget.
        // - leaf agg: Poseidon circuit has PI = [state]
        // - internal agg: child agg circuits have PI = [Hash(inner_vk_repr, state, acc_pi, level)]
        let (assigned_left_pi, assigned_right_pi) = if self.is_leaf {
            // For leaf aggregation, we want the provided PI accumulators to be "available for bases"
            // but not contribute to the MSM: scale them to neutral by multiplying by 0.
            let neutral_scaling_factor = scalar_chip.assign_fixed(&mut layouter, false)?;

            AssignedAccumulator::scale_by_bit(
                &mut layouter,
                &scalar_chip,
                &neutral_scaling_factor,
                &mut left_acc,
            )?;
            AssignedAccumulator::scale_by_bit(
                &mut layouter,
                &scalar_chip,
                &neutral_scaling_factor,
                &mut right_acc,
            )?;

            left_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;
            right_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

            (vec![left_state.clone()], vec![right_state.clone()])
        } else {
            let inner_vk_pi: AssignedNative<F> =
                native_chip.assign_fixed(&mut layouter, self.inner_vk_repr)?;

            let mut left_pi_vec = vec![inner_vk_pi.clone()];
            left_pi_vec.push(left_state.clone());
            left_pi_vec.extend(verifier_chip.as_public_input(&mut layouter, &left_acc)?);
            left_pi_vec.push(prev_level.clone());
            let left_hash = poseidon_chip.hash(&mut layouter, &left_pi_vec)?;
            let assigned_left_pi = vec![left_hash];

            let mut right_pi_vec = vec![inner_vk_pi];
            right_pi_vec.push(right_state.clone());
            right_pi_vec.extend(verifier_chip.as_public_input(&mut layouter, &right_acc)?);
            right_pi_vec.push(prev_level.clone());
            let right_hash = poseidon_chip.hash(&mut layouter, &right_pi_vec)?;
            let assigned_right_pi = vec![right_hash];

            (assigned_left_pi, assigned_right_pi)
        };

        let id_point: AssignedForeignPoint<
            midnight_curves::Fq,
            midnight_curves::G1Projective,
            midnight_curves::G1Projective,
        > = curve_chip.assign_fixed(&mut layouter, C::identity())?;

        // Process left child
        let mut left_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_vk,
            &[("com_instance", id_point.clone())],
            &[&assigned_left_pi],
            self.left_proof.clone(),
        )?;
        left_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        // Process right child
        let mut right_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_vk,
            &[("com_instance", id_point)],
            &[&assigned_right_pi],
            self.right_proof.clone(),
        )?;
        right_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        // Accumulate and output
        let mut next_acc = AssignedAccumulator::<S>::accumulate(
            &mut layouter,
            &verifier_chip,
            &scalar_chip,
            &poseidon_chip,
            &[left_proof_acc, left_acc, right_proof_acc, right_acc],
        )?;
        next_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        // Hash the would-be public inputs (vk_for_pi, state, acc, level) into a single public input
        // where vk_for_pi for this circuit is the *child VK repr*.
        let next_acc_pi = verifier_chip.as_public_input(&mut layouter, &next_acc)?;
        let mut hash_inputs = Vec::new();
        hash_inputs.push(child_vk_val);
        hash_inputs.push(next_state);
        hash_inputs.extend(next_acc_pi);
        hash_inputs.push(next_level);
        let agg_pi_hash = poseidon_chip.hash(&mut layouter, &hash_inputs)?;

        // Expose only the hash as a public input of the aggregation circuit
        native_chip.constrain_as_public_input(&mut layouter, &agg_pi_hash)?;

        core_decomp_chip.load(&mut layouter)
    }
}

#[derive(Clone, Debug)]
struct TreeNode {
    state: F,
    proof: Vec<u8>,
    proof_acc: Accumulator<S>,
    pi_acc: Accumulator<S>,
}

fn fixed_base_names_for(vk_name: &str, cs: &ConstraintSystem<F>) -> Vec<String> {
    let mut names = vec![String::from("com_instance"), String::from("~G")];
    names.extend(midnight_circuits::verifier::fixed_base_names::<S>(
        vk_name,
        cs.num_fixed_columns() + cs.num_selectors(),
        cs.permutation().columns.len(),
    ));
    names
}

fn trivial_acc_with_names(names: &[String]) -> Accumulator<S> {
    use midnight_circuits::verifier::Msm;
    let fixed: BTreeMap<String, F> = names.iter().cloned().map(|n| (n, F::ZERO)).collect();
    Accumulator::<S>::new(
        Msm::new(&[C::default()], &[F::ONE], &BTreeMap::new()),
        Msm::new(&[C::default()], &[F::ONE], &fixed),
    )
}

fn poseidon_tree_root(leaf_states: &[F]) -> F {
    use midnight_circuits::instructions::hash::HashCPU;
    assert!(!leaf_states.is_empty() && leaf_states.len().is_power_of_two());

    let mut level_states = leaf_states.to_vec();
    while level_states.len() > 1 {
        level_states = level_states
            .chunks(2)
            .map(|pair| <PoseidonChip<F> as HashCPU<F, F>>::hash(&[pair[0], pair[1]]))
            .collect();
    }
    level_states[0]
}

fn verify_and_extract_acc(
    srs: &ParamsKZG<Bls12>,
    vk: &VerifyingKey<F, KZGCommitmentScheme<E>>,
    fixed_bases: &BTreeMap<String, C>,
    proof: &[u8],
    public_inputs: &[F],
) -> Accumulator<S> {
    use midnight_proofs::{plonk::prepare, transcript::CircuitTranscript};

    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);
    let dual_msm = prepare::<F, KZGCommitmentScheme<E>, CircuitTranscript<PoseidonState<F>>>(
        vk,
        &[&[C::identity()]],
        &[&[public_inputs]],
        &mut transcript,
    )
    .expect("Verification failed");

    assert!(dual_msm.clone().check(&srs.verifier_params()));

    let mut acc: Accumulator<S> = dual_msm.into();
    acc.extract_fixed_bases(fixed_bases);
    acc.collapse();
    assert!(acc.check(&srs.s_g2().into(), fixed_bases));
    acc
}

// Off-circuit helper to hash the would-be public inputs (vk_for_pi, state, acc, level)
// into a single field element that is used as the only public input of AggCircuit.
fn agg_public_input_hash(vk_repr: F, state: F, acc: &Accumulator<S>, level: F) -> F {
    use midnight_circuits::instructions::hash::HashCPU;

    let mut inputs = Vec::new();
    inputs.push(vk_repr);
    inputs.push(state);
    inputs.extend(AssignedAccumulator::as_public_input(acc));
    inputs.push(level);

    <PoseidonChip<F> as HashCPU<F, F>>::hash(&inputs)
}

fn agg_vk_name_for_level(level: usize) -> String {
    // Level 1 is the leaf-aggregation circuit (verifies Poseidon proofs).
    // Level >=2 are internal aggregation circuits (each verifies the previous level circuit).
    format!("agg_vk_lvl{}", level)
}

fn main() {
    // -----------------------------
    // Parameters / tree shape
    // -----------------------------
    let num_leaves: usize = 8;
    assert!(num_leaves.is_power_of_two());
    let max_level: usize = (num_leaves as u32).trailing_zeros() as usize; // e.g. 8 -> 3 levels

    // -----------------------------
    // Setup Poseidon circuit
    // -----------------------------
    let poseidon_srs = filecoin_srs(POSEIDON_K);
    let poseidon_relation = PoseidonExample;
    let poseidon_vk = compact_std_lib::setup_vk(&poseidon_srs, &poseidon_relation);
    let poseidon_pk = compact_std_lib::setup_pk(&poseidon_relation, &poseidon_vk);
    let poseidon_halo2_vk = poseidon_vk.vk();
    let poseidon_vk_data: (EvaluationDomain<F>, ConstraintSystem<F>, F) = (
        EvaluationDomain::new(poseidon_halo2_vk.cs().degree() as u32, POSEIDON_K),
        poseidon_halo2_vk.cs().clone(),
        poseidon_halo2_vk.transcript_repr(),
    );

    let mut poseidon_fixed_bases = BTreeMap::new();
    poseidon_fixed_bases.insert(String::from("com_instance"), C::identity());
    poseidon_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "poseidon_vk",
        poseidon_halo2_vk,
    ));

    // -----------------------------
    // Setup aggregation CS/domain and SRS
    // -----------------------------
    let mut agg_cs = ConstraintSystem::default();
    configure_agg_circuit(&mut agg_cs);
    let agg_domain: EvaluationDomain<F> = EvaluationDomain::new(agg_cs.degree() as u32, K);
    let agg_srs = filecoin_srs(K);

    // -----------------------------
    // Precompute all AGG vk names (one per recursion layer)
    // -----------------------------
    let agg_vk_names: Vec<String> = (1..=max_level).map(agg_vk_name_for_level).collect();

    // -----------------------------
    // Build a *global* fixed base name list that includes Poseidon + every AGG level vk_name.
    // This avoids collisions because each level has a distinct vk_name prefix.
    // -----------------------------
    let combined_fixed_base_names: Vec<String> = {
        let mut set = BTreeSet::new();
        let mut out = Vec::new();

        for name in fixed_base_names_for("poseidon_vk", &poseidon_vk_data.1) {
            if set.insert(name.clone()) {
                out.push(name);
            }
        }

        for vk_name in agg_vk_names.iter() {
            for name in fixed_base_names_for(vk_name.as_str(), &agg_cs) {
                if set.insert(name.clone()) {
                    out.push(name);
                }
            }
        }

        out
    };

    // -----------------------------
    // Keygen one AGG VK/PK per level, each using a distinct vk_name (=> distinct fixed-base keys)
    // -----------------------------
    let mut agg_vks: Vec<Option<Arc<VerifyingKey<F, KZGCommitmentScheme<E>>>>> =
        vec![None; max_level + 1];
    let mut agg_pks: Vec<Option<Arc<_>>> = vec![None; max_level + 1];
    let mut agg_vk_data: Vec<Option<(EvaluationDomain<F>, ConstraintSystem<F>, F)>> =
        vec![None; max_level + 1];

    for level in 1..=max_level {
        let (child_vk, child_vk_name, inner_vk_repr, expected_prev_level, is_leaf) = if level == 1 {
            (
                poseidon_vk_data.clone(),
                "poseidon_vk".to_string(),
                F::ZERO,
                F::ZERO,
                true,
            )
        } else {
            let child = agg_vk_data[level - 1]
                .as_ref()
                .expect("child agg vk_data missing")
                .clone();
            let child_name = agg_vk_names[level - 2].clone(); // level-1 name is at index (level-2)

            // inner_vk_repr is what the *child* circuit embedded as vk_for_pi in its own PI hash:
            // - if child is level 1, it's poseidon_vk repr
            // - if child is level >=2, it's the transcript repr of (level-2) circuit
            let inner = if level == 2 {
                poseidon_vk_data.2
            } else {
                agg_vk_data[level - 2]
                    .as_ref()
                    .expect("grandchild agg vk_data missing")
                    .2
            };

            (child, child_name, inner, F::from((level as u64) - 1), false)
        };

        let default_circuit = AggCircuit {
            child_vk,
            child_vk_name,
            inner_vk_repr,
            expected_prev_level,
            left_state: Value::unknown(),
            right_state: Value::unknown(),
            left_proof: Value::unknown(),
            right_proof: Value::unknown(),
            left_acc: Value::unknown(),
            right_acc: Value::unknown(),
            fixed_base_names: combined_fixed_base_names.clone(),
            prev_level: Value::unknown(),
            is_leaf,
        };

        let start = Instant::now();
        let vk = keygen_vk_with_k(&agg_srs, &default_circuit, K).unwrap();
        let pk = keygen_pk(vk.clone(), &default_circuit).unwrap();

        println!(
            "Computed {} vk/pk in {:?}",
            agg_vk_names[level - 1],
            start.elapsed()
        );

        let vk_data = (
            EvaluationDomain::new(vk.cs().degree() as u32, K),
            vk.cs().clone(),
            vk.transcript_repr(),
        );

        agg_vks[level] = Some(Arc::new(vk));
        agg_pks[level] = Some(Arc::new(pk));
        agg_vk_data[level] = Some(vk_data);
    }

    // -----------------------------
    // Build fixed-base maps per circuit (Poseidon + each AGG level) and a combined map
    // -----------------------------
    let mut fixed_bases_by_level: Vec<Option<BTreeMap<String, C>>> = vec![None; max_level + 1];

    for level in 1..=max_level {
        let vk_name = agg_vk_names[level - 1].as_str();
        let vk = agg_vks[level].as_ref().unwrap();

        let mut bases = BTreeMap::new();
        bases.insert(String::from("com_instance"), C::identity());
        bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
            vk_name,
            vk.as_ref(),
        ));
        fixed_bases_by_level[level] = Some(bases);
    }

    let mut combined_fixed_bases = BTreeMap::new();
    for (name, base) in poseidon_fixed_bases.iter() {
        combined_fixed_bases.insert(name.clone(), *base);
    }
    for level in 1..=max_level {
        for (name, base) in fixed_bases_by_level[level].as_ref().unwrap().iter() {
            combined_fixed_bases.insert(name.clone(), *base);
        }
    }

    // -----------------------------
    // Build a "global trivial" accumulator carrying fixed bases for *all* circuits (no collisions)
    // -----------------------------
    let trivial_poseidon =
        trivial_acc_with_names(&fixed_base_names_for("poseidon_vk", &poseidon_vk_data.1));

    let mut trivial_all: Vec<Accumulator<S>> = vec![trivial_poseidon];
    for level in 1..=max_level {
        let vk_name = agg_vk_names[level - 1].as_str();
        let cs = &agg_vks[level].as_ref().unwrap().cs();
        let t = trivial_acc_with_names(&fixed_base_names_for(vk_name, cs));
        trivial_all.push(t);
    }

    let mut trivial_combined = Accumulator::accumulate(&trivial_all);
    trivial_combined.collapse();

    // -----------------------------
    // Generate Poseidon leaf proofs
    // -----------------------------
    println!("Creating {} POSEIDON leaf proofs...", num_leaves);

    let poseidon_proofs: Vec<(F, [F; 3], Vec<u8>)> = (0..num_leaves)
        .map(|i| {
            use midnight_circuits::instructions::hash::HashCPU;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
            let witness: [F; 3] = core::array::from_fn(|_| F::random(&mut rng));
            let instance = <PoseidonChip<F> as HashCPU<F, F>>::hash(&witness);

            let proof = {
                let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<E>,
                    CircuitTranscript<PoseidonState<F>>,
                    MidnightCircuit<PoseidonExample>,
                >(
                    &poseidon_srs,
                    &poseidon_pk.pk(),
                    &[MidnightCircuit::new(
                        &poseidon_relation,
                        Value::known(instance),
                        Value::known(witness),
                        Some(1),
                    )],
                    1,
                    &[&[&[], &[instance]]],
                    OsRng,
                    &mut transcript,
                )
                .expect("Poseidon proof failed");
                transcript.finalize()
            };

            println!("POSEIDON leaf {} created", i);
            (instance, witness, proof)
        })
        .collect();

    // -----------------------------
    // Create leaf aggregation layer (AGG level 1, verifies Poseidon proofs)
    // -----------------------------
    println!("\nCreating {} leaf AGG nodes...", num_leaves / 2);

    let leaf_agg_vk_name = agg_vk_names[0].clone(); // lvl1
    let leaf_agg_vk = agg_vks[1].as_ref().unwrap().clone();
    let leaf_agg_pk = agg_pks[1].as_ref().unwrap().clone();
    let leaf_agg_fixed_bases = fixed_bases_by_level[1].as_ref().unwrap();

    let mut current_level: Vec<TreeNode> = (0..num_leaves / 2)
        .map(|i| {
            use midnight_circuits::instructions::hash::HashCPU;
            let (left_state, _, left_proof) = &poseidon_proofs[i * 2];
            let (right_state, _, right_proof) = &poseidon_proofs[i * 2 + 1];

            let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[*left_state, *right_state]);

            let circuit = AggCircuit {
                child_vk: poseidon_vk_data.clone(),
                child_vk_name: "poseidon_vk".to_string(),
                inner_vk_repr: F::ZERO,
                expected_prev_level: F::ZERO,
                left_state: Value::known(*left_state),
                right_state: Value::known(*right_state),
                left_proof: Value::known(left_proof.clone()),
                right_proof: Value::known(right_proof.clone()),
                left_acc: Value::known(trivial_combined.clone()),
                right_acc: Value::known(trivial_combined.clone()),
                fixed_base_names: combined_fixed_base_names.clone(),
                is_leaf: true,
                prev_level: Value::known(F::ZERO),
            };

            let proof_acc_left = verify_and_extract_acc(
                &poseidon_srs,
                poseidon_vk.vk(),
                &poseidon_fixed_bases,
                left_proof,
                &[*left_state],
            );
            let proof_acc_right = verify_and_extract_acc(
                &poseidon_srs,
                poseidon_vk.vk(),
                &poseidon_fixed_bases,
                right_proof,
                &[*right_state],
            );

            let mut accumulated_pi = Accumulator::accumulate(&[
                proof_acc_left,
                trivial_combined.clone(),
                proof_acc_right,
                trivial_combined.clone(),
            ]);
            accumulated_pi.collapse();

            // Public input hash for level 1 uses vk_for_pi = poseidon vk repr, level = 1
            let public_input_hash =
                agg_public_input_hash(poseidon_vk_data.2, state, &accumulated_pi, F::ONE);
            let public_inputs = vec![public_input_hash];

            let start = Instant::now();
            let proof = {
                let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<E>,
                    CircuitTranscript<PoseidonState<F>>,
                    AggCircuit,
                >(
                    &agg_srs,
                    leaf_agg_pk.as_ref(),
                    &[circuit],
                    1,
                    &[&[&[], &public_inputs]],
                    OsRng,
                    &mut transcript,
                )
                .expect("Leaf AGG proof failed");
                transcript.finalize()
            };
            println!(
                "Leaf AGG {} ({}) created in {:?}",
                i,
                leaf_agg_vk_name,
                start.elapsed()
            );

            let proof_acc = verify_and_extract_acc(
                &agg_srs,
                leaf_agg_vk.as_ref(),
                leaf_agg_fixed_bases,
                &proof,
                &public_inputs,
            );

            assert!(accumulated_pi.check(&agg_srs.s_g2().into(), &combined_fixed_bases));

            TreeNode {
                state,
                proof,
                proof_acc,
                pi_acc: accumulated_pi,
            }
        })
        .collect();

    // -----------------------------
    // Build internal layers: each layer L uses its own vk_name and verifies layer L-1
    // -----------------------------
    let mut child_level: usize = 1; // current nodes are at AGG level 1
    while current_level.len() > 1 {
        let parent_level = child_level + 1;
        let parent_vk_name = agg_vk_names[parent_level - 1].clone();
        println!(
            "\nBuilding AGG level {} ({}) with {} nodes...",
            parent_level,
            parent_vk_name,
            current_level.len() / 2
        );

        let parent_vk = agg_vks[parent_level].as_ref().unwrap().clone();
        let parent_pk = agg_pks[parent_level].as_ref().unwrap().clone();
        let parent_fixed_bases = fixed_bases_by_level[parent_level].as_ref().unwrap();

        let child_vk_data = agg_vk_data[child_level].as_ref().unwrap().clone();
        let child_vk_repr = child_vk_data.2;
        let child_vk_name = agg_vk_names[child_level - 1].clone();

        // inner_vk_repr is what the *child* circuit embedded as vk_for_pi
        let inner_vk_repr = if child_level == 1 {
            // child is level 1 => it verified poseidon
            poseidon_vk_data.2
        } else {
            // child is >=2 => it verified (child_level-1)
            agg_vk_data[child_level - 1].as_ref().unwrap().2
        };

        let next_level: Vec<TreeNode> = (0..current_level.len() / 2)
            .map(|i| {
                use midnight_circuits::instructions::hash::HashCPU;
                let left = &current_level[i * 2];
                let right = &current_level[i * 2 + 1];

                let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

                let circuit = AggCircuit {
                    child_vk: child_vk_data.clone(),
                    child_vk_name: child_vk_name.clone(),
                    inner_vk_repr,
                    expected_prev_level: F::from(child_level as u64),
                    left_state: Value::known(left.state),
                    right_state: Value::known(right.state),
                    left_proof: Value::known(left.proof.clone()),
                    right_proof: Value::known(right.proof.clone()),
                    left_acc: Value::known(left.pi_acc.clone()),
                    right_acc: Value::known(right.pi_acc.clone()),
                    fixed_base_names: combined_fixed_base_names.clone(),
                    is_leaf: false,
                    prev_level: Value::known(F::from(child_level as u64)),
                };

                let mut accumulated_pi = Accumulator::accumulate(&[
                    left.proof_acc.clone(),
                    left.pi_acc.clone(),
                    right.proof_acc.clone(),
                    right.pi_acc.clone(),
                ]);
                accumulated_pi.collapse();

                // Public input hash for this parent node:
                //   vk_for_pi = transcript repr of *child* circuit (level = child_level)
                //   level     = parent_level
                let public_input_hash = agg_public_input_hash(
                    child_vk_repr,
                    state,
                    &accumulated_pi,
                    F::from(parent_level as u64),
                );
                let public_inputs = vec![public_input_hash];

                let start = Instant::now();
                let proof = {
                    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                    create_proof::<
                        F,
                        KZGCommitmentScheme<E>,
                        CircuitTranscript<PoseidonState<F>>,
                        AggCircuit,
                    >(
                        &agg_srs,
                        parent_pk.as_ref(),
                        &[circuit],
                        1,
                        &[&[&[], &public_inputs]],
                        OsRng,
                        &mut transcript,
                    )
                    .expect("Internal AGG proof failed");
                    transcript.finalize()
                };
                println!(
                    "Level {} node {} ({}) created in {:?}",
                    parent_level,
                    i,
                    parent_vk_name,
                    start.elapsed()
                );

                assert!(accumulated_pi.check(&agg_srs.s_g2().into(), &combined_fixed_bases));

                let proof_acc = verify_and_extract_acc(
                    &agg_srs,
                    parent_vk.as_ref(),
                    parent_fixed_bases,
                    &proof,
                    &public_inputs,
                );

                TreeNode {
                    state,
                    proof,
                    proof_acc,
                    pi_acc: accumulated_pi,
                }
            })
            .collect();

        current_level = next_level;
        child_level = parent_level;
    }

    let root = &current_level[0];
    println!("\n=== AGG Tree Complete ===");
    println!("Root state: {:?}", root.state);

    let leaf_states: Vec<F> = poseidon_proofs.iter().map(|(s, _, _)| *s).collect();
    let expected_root = poseidon_tree_root(&leaf_states);

    println!(
        "Expected root (recomputed from POSEIDON proofs): {:?}",
        expected_root
    );
    assert_eq!(root.state, expected_root, "Root state mismatch!");
    println!("✓ Root verification successful!");
}
