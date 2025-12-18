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
        prepare,
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
    agg_vk: (EvaluationDomain<F>, ConstraintSystem<F>, Value<F>),
    agg_vk_name: &'static str,
    poseidon_vk: (EvaluationDomain<F>, ConstraintSystem<F>, F),
    leaf_agg_vk: (EvaluationDomain<F>, ConstraintSystem<F>, Value<F>),
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

        // Assign and compute level information
        let prev_level = scalar_chip.assign(&mut layouter, self.prev_level)?;
        let next_level = scalar_chip.add_constant(&mut layouter, &prev_level, F::ONE)?;
        let is_genesis = scalar_chip.is_equal_to_fixed(&mut layouter, &prev_level, F::ZERO)?;
        let children_are_genesis =
            scalar_chip.is_equal_to_fixed(&mut layouter, &prev_level, F::ONE)?;
        let is_level_2 =
            scalar_chip.is_equal_to_fixed(&mut layouter, &prev_level, F::from(2u64))?;

        // Poseidon vk transcript repr is a constant in both circuits
        let poseidon_vk_val: AssignedNative<F> =
            native_chip.assign_fixed(&mut layouter, self.poseidon_vk.2)?;

        // Compute next state (Poseidon over children states)
        let left_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.left_state)?;
        let right_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.right_state)?;
        let next_state =
            poseidon_chip.hash(&mut layouter, &[left_state.clone(), right_state.clone()])?;

        let id_point: AssignedForeignPoint<
            midnight_curves::Fq,
            midnight_curves::G1Projective,
            midnight_curves::G1Projective,
        > = curve_chip.assign_fixed(&mut layouter, C::identity())?;

        // Assigned accumulators for children
        let left_acc = AssignedAccumulator::assign(
            &mut layouter,
            &curve_chip,
            &scalar_chip,
            1,
            1,
            &[],
            &self.fixed_base_names,
            self.left_acc.clone(),
        )?;
        let right_acc = AssignedAccumulator::assign(
            &mut layouter,
            &curve_chip,
            &scalar_chip,
            1,
            1,
            &[],
            &self.fixed_base_names,
            self.right_acc.clone(),
        )?;

        // VK selection and inner public inputs are handled differently
        // for the leaf aggregation vs recursive aggregation circuits.
        let (assigned_vk, vk_for_pi, assigned_left_pi, assigned_right_pi) = if self.is_leaf {
            // -------------------------------------------------------------
            // Leaf aggregation circuit: verify Poseidon proofs with poseidon_vk
            // -------------------------------------------------------------
            let assigned_vk = verifier_chip.assign_vk(
                self.agg_vk_name,
                &self.poseidon_vk.0,
                &self.poseidon_vk.1,
                poseidon_vk_val.clone(),
            )?;

            // Poseidon circuit has a single public input: its output state.
            let assigned_left_pi = vec![left_state.clone()];
            let assigned_right_pi = vec![right_state.clone()];

            (
                assigned_vk,
                poseidon_vk_val.clone(),
                assigned_left_pi,
                assigned_right_pi,
            )
        } else {
            // -------------------------------------------------------------
            // Recursive aggregation circuit:
            //   - leaf_agg vk is a constant in-circuit
            //   - self-recursive agg vk is a witness
            //   - we select in-circuit between poseidon / leaf_agg / agg vk
            // -------------------------------------------------------------
            let mut leaf_agg_vk_repr = F::ZERO;
            self.leaf_agg_vk.2.map(|v| leaf_agg_vk_repr = v);
            let leaf_vk_val: AssignedNative<F> =
                native_chip.assign_fixed(&mut layouter, leaf_agg_vk_repr)?;

            let agg_vk_val: AssignedNative<F> = native_chip.assign(&mut layouter, self.agg_vk.2)?;

            // Select VK used by the partial verifier based on the current level:
            //   prev_level == 0 -> poseidon vk
            //   prev_level == 1 -> leaf_agg vk
            //   prev_level >= 2 -> self-recursive agg vk
            let vk_level_sel = native_chip.select(
                &mut layouter,
                &children_are_genesis,
                &leaf_vk_val,
                &agg_vk_val,
            )?;
            let vk_val =
                native_chip.select(&mut layouter, &is_genesis, &poseidon_vk_val, &vk_level_sel)?;

            // The assigned_vk is used in the partial verification gadget and
            // directly affects the circuit shape. For the recursive agg case
            // we feed in the in-circuit-selected vk representation.
            let assigned_vk = verifier_chip.assign_vk(
                self.agg_vk_name,
                &self.agg_vk.0,
                &self.agg_vk.1,
                vk_val.clone(),
            )?;

            // Select inner VK public inputs based on the level "below":
            //   prev_level == 1 -> poseidon vk
            //   prev_level == 2 -> leaf_agg vk
            //   prev_level >= 3 -> agg vk
            let vk_inner_tmp = native_chip.select(
                &mut layouter,
                &children_are_genesis,
                &poseidon_vk_val,
                &agg_vk_val,
            )?;
            let vk_inner_pi =
                native_chip.select(&mut layouter, &is_level_2, &leaf_vk_val, &vk_inner_tmp)?;

            // Hash the child public inputs into a single field element
            // which becomes the inner PI for this layer.
            let mut left_pi_vec = vec![vk_inner_pi.clone()];
            left_pi_vec.push(left_state.clone());
            left_pi_vec.extend(verifier_chip.as_public_input(&mut layouter, &left_acc)?);
            left_pi_vec.push(prev_level.clone());
            let left_hash = poseidon_chip.hash(&mut layouter, &left_pi_vec)?;
            let assigned_left_pi = vec![left_hash];

            let mut right_pi_vec = vec![vk_inner_pi];
            right_pi_vec.push(right_state.clone());
            right_pi_vec.extend(verifier_chip.as_public_input(&mut layouter, &right_acc)?);
            right_pi_vec.push(prev_level.clone());
            let right_hash = poseidon_chip.hash(&mut layouter, &right_pi_vec)?;
            let assigned_right_pi = vec![right_hash];

            (assigned_vk, vk_val, assigned_left_pi, assigned_right_pi)
        };

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

        // Hash the would-be public inputs (vk, state, acc, level) into a single public input
        let next_acc_pi = verifier_chip.as_public_input(&mut layouter, &next_acc)?;
        let mut hash_inputs = Vec::new();
        hash_inputs.push(vk_for_pi);
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

// Off-circuit helper to hash the would-be public inputs (vk, state, acc, level)
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

fn main() {
    // Setup Poseidon circuit
    let poseidon_srs = filecoin_srs(POSEIDON_K);
    let poseidon_relation = PoseidonExample;
    let poseidon_vk = compact_std_lib::setup_vk(&poseidon_srs, &poseidon_relation);
    let poseidon_pk = compact_std_lib::setup_pk(&poseidon_relation, &poseidon_vk);
    let poseidon_halo2_vk = poseidon_vk.vk();
    let poseidon_vk_data = (
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

    // Setup aggregation circuit constraint system
    let mut agg_cs = ConstraintSystem::default();
    configure_agg_circuit(&mut agg_cs);
    let agg_domain = EvaluationDomain::new(agg_cs.degree() as u32, K);

    let combined_fixed_base_names_keygen: Vec<String> = {
        let mut set = BTreeSet::new();
        let mut v = Vec::new();
        for name in fixed_base_names_for("poseidon_vk", &poseidon_vk_data.1)
            .iter()
            .chain(fixed_base_names_for("agg_vk", &agg_cs).iter())
        {
            if set.insert(name.clone()) {
                v.push(name.clone());
            }
        }
        v
    };

    let agg_srs = filecoin_srs(K);

    // First: leaf aggregation circuit (verifies Poseidon proofs with poseidon_vk)
    let default_leaf_agg_circuit = AggCircuit {
        agg_vk: (agg_domain.clone(), agg_cs.clone(), Value::unknown()),
        agg_vk_name: "poseidon_vk",
        poseidon_vk: poseidon_vk_data.clone(),
        left_state: Value::unknown(),
        right_state: Value::unknown(),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_acc: Value::unknown(),
        right_acc: Value::unknown(),
        fixed_base_names: combined_fixed_base_names_keygen.clone(),
        is_leaf: true,
        prev_level: Value::unknown(),
        leaf_agg_vk: (agg_domain.clone(), agg_cs.clone(), Value::unknown()),
    };

    let leaf_agg_vk = keygen_vk_with_k(&agg_srs, &default_leaf_agg_circuit, K).unwrap();
    let leaf_agg_pk = keygen_pk(leaf_agg_vk.clone(), &default_leaf_agg_circuit).unwrap();
    println!("Computed leaf AGG vk and pk");

    let leaf_agg_vk_data = (
        EvaluationDomain::new(leaf_agg_vk.cs().degree() as u32, K),
        leaf_agg_vk.cs().clone(),
        Value::known(leaf_agg_vk.transcript_repr()),
    );

    // Now: recursive aggregation circuit (can aggregate leaf_agg or itself)
    let default_agg_circuit = AggCircuit {
        agg_vk: (agg_domain.clone(), agg_cs.clone(), Value::unknown()),
        agg_vk_name: "agg_vk",
        poseidon_vk: poseidon_vk_data.clone(),
        left_state: Value::unknown(),
        right_state: Value::unknown(),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_acc: Value::unknown(),
        right_acc: Value::unknown(),
        fixed_base_names: combined_fixed_base_names_keygen.clone(),
        is_leaf: false,
        prev_level: Value::unknown(),
        leaf_agg_vk: leaf_agg_vk_data.clone(),
    };

    let start = Instant::now();
    let agg_vk = keygen_vk_with_k(&agg_srs, &default_agg_circuit, K).unwrap();
    let agg_pk = keygen_pk(agg_vk.clone(), &default_agg_circuit).unwrap();
    println!("Computed AGG vk and pk in {:?}", start.elapsed());

    let mut agg_fixed_bases = BTreeMap::new();
    agg_fixed_bases.insert(String::from("com_instance"), C::identity());
    agg_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "agg_vk", &agg_vk,
    ));

    let mut leaf_agg_fixed_bases = BTreeMap::new();
    leaf_agg_fixed_bases.insert(String::from("com_instance"), C::identity());
    leaf_agg_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "agg_vk",
        &leaf_agg_vk,
    ));

    let agg_srs = Arc::new(agg_srs);
    let agg_vk = Arc::new(agg_vk);
    let agg_pk = Arc::new(agg_pk);

    let combined_fixed_base_names: Vec<String> = {
        let mut set = BTreeSet::new();
        let mut v = Vec::new();
        for name in fixed_base_names_for("poseidon_vk", &poseidon_vk_data.1)
            .iter()
            .chain(fixed_base_names_for("agg_vk", &leaf_agg_vk.cs()).iter())
            .chain(fixed_base_names_for("agg_vk", &agg_vk.cs()).iter())
        {
            if set.insert(name.clone()) {
                v.push(name.clone());
            }
        }
        v
    };

    let trivial_poseidon =
        trivial_acc_with_names(&fixed_base_names_for("poseidon_vk", &poseidon_vk_data.1));
    let trivial_leaf_agg =
        trivial_acc_with_names(&fixed_base_names_for("agg_vk", &leaf_agg_vk.cs()));
    let trivial_agg = trivial_acc_with_names(&fixed_base_names_for("agg_vk", &agg_vk.cs()));

    let mut trivial_combined =
        Accumulator::accumulate(&[trivial_poseidon, trivial_leaf_agg, trivial_agg]);
    trivial_combined.collapse();

    let agg_vk_data = (
        EvaluationDomain::new(agg_vk.cs().degree() as u32, K),
        agg_vk.cs().clone(),
        Value::known(agg_vk.transcript_repr()),
    );

    // Generate leaf proofs
    let num_leaves = 8;
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

    // Create leaf aggregation layer
    println!("\nCreating {} leaf AGG nodes...", num_leaves / 2);

    let mut current_level: Vec<TreeNode> = (0..num_leaves / 2)
        .map(|i| {
            use midnight_circuits::instructions::hash::HashCPU;
            let (left_state, _, left_proof) = &poseidon_proofs[i * 2];
            let (right_state, _, right_proof) = &poseidon_proofs[i * 2 + 1];

            let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[*left_state, *right_state]);

            let circuit = AggCircuit {
                agg_vk: agg_vk_data.clone(),
                agg_vk_name: "poseidon_vk",
                poseidon_vk: poseidon_vk_data.clone(),
                left_state: Value::known(*left_state),
                right_state: Value::known(*right_state),
                left_proof: Value::known(left_proof.clone()),
                right_proof: Value::known(right_proof.clone()),
                left_acc: Value::known(trivial_combined.clone()),
                right_acc: Value::known(trivial_combined.clone()),
                fixed_base_names: combined_fixed_base_names.clone(),
                is_leaf: true,
                prev_level: Value::known(F::ZERO),
                leaf_agg_vk: leaf_agg_vk_data.clone(),
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

            // Hash (vk, state, acc, level) into a single public input for the leaf agg circuit
            let vk_repr = poseidon_halo2_vk.transcript_repr();
            let public_input_hash = agg_public_input_hash(vk_repr, state, &accumulated_pi, F::ONE);
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
                    &leaf_agg_pk,
                    &[circuit],
                    1,
                    &[&[&[], &public_inputs]],
                    OsRng,
                    &mut transcript,
                )
                .expect("Leaf AGG proof failed");
                transcript.finalize()
            };
            println!("Leaf AGG {} created in {:?}", i, start.elapsed());

            let proof_acc = verify_and_extract_acc(
                &agg_srs,
                &leaf_agg_vk,
                &leaf_agg_fixed_bases,
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

    // Build internal layers
    let mut level = 0;
    while current_level.len() > 1 {
        level += 1;
        println!(
            "\nBuilding AGG level {} with {} nodes...",
            level,
            current_level.len() / 2
        );

        let next_level: Vec<TreeNode> = (0..current_level.len() / 2)
            .map(|i| {
                use midnight_circuits::instructions::hash::HashCPU;
                let left = &current_level[i * 2];
                let right = &current_level[i * 2 + 1];

                let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

                let circuit = AggCircuit {
                    agg_vk: agg_vk_data.clone(),
                    agg_vk_name: "agg_vk",
                    poseidon_vk: poseidon_vk_data.clone(),
                    left_state: Value::known(left.state),
                    right_state: Value::known(right.state),
                    left_proof: Value::known(left.proof.clone()),
                    right_proof: Value::known(right.proof.clone()),
                    left_acc: Value::known(left.pi_acc.clone()),
                    right_acc: Value::known(right.pi_acc.clone()),
                    fixed_base_names: combined_fixed_base_names.clone(),
                    is_leaf: false,
                    prev_level: Value::known(F::from(level)),
                    leaf_agg_vk: leaf_agg_vk_data.clone(),
                };

                let mut accumulated_pi = Accumulator::accumulate(&[
                    left.proof_acc.clone(),
                    left.pi_acc.clone(),
                    right.proof_acc.clone(),
                    right.pi_acc.clone(),
                ]);
                accumulated_pi.collapse();

                let input_vk = if level == 1 {
                    &leaf_agg_vk
                } else {
                    agg_vk.as_ref()
                };

                // Hash (vk, state, acc, level+1) into a single public input for internal agg nodes
                let vk_repr = input_vk.transcript_repr();
                let next_level_f = F::from(level + 1);
                let public_input_hash =
                    agg_public_input_hash(vk_repr, state, &accumulated_pi, next_level_f);
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
                        &agg_pk,
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
                    "Level {} node {} created in {:?}",
                    level,
                    i,
                    start.elapsed()
                );

                let proof_acc = verify_and_extract_acc(
                    &agg_srs,
                    &agg_vk,
                    &agg_fixed_bases,
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
