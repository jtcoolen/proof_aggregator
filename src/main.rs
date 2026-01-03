// NOTE: Option B (NUMA-aware pools) requires adding these deps to Cargo.toml:
//   crossbeam-channel = "0.5"
//   core_affinity     = "0.8"
// (rayon is already used)

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
        Circuit, ConstraintSystem, Error, ProvingKey, VerifyingKey, create_proof, keygen_pk,
        keygen_vk_with_k,
    },
    poly::{EvaluationDomain, kzg::KZGCommitmentScheme},
    transcript::{CircuitTranscript, Transcript},
};
use rand::rngs::OsRng;

use core_affinity::CoreId;
use crossbeam_channel as channel;
use rayon::{ThreadPool, ThreadPoolBuilder};

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

type S = BlstrsEmulation;
type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;
type E = <S as SelfEmulation>::Engine;
type CBase = <C as CircuitCurve>::Base;
type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

type Vk = VerifyingKey<F, KZGCommitmentScheme<E>>;
type Pk = ProvingKey<F, KZGCommitmentScheme<E>>;

const K: u32 = 19;
const K2: u32 = 20;
const POSEIDON_K: u32 = 6;

// -----------------------------
// Option B (adaptive): pool sets for 8/4/2/1 concurrent proofs.
// Your machine: 4 NUMA nodes × 36 cores each (per lscpu)
// -----------------------------
const NUMA_NODES: usize = 4;
const CORES_PER_NUMA: usize = 36;
const TOTAL_CORES: usize = NUMA_NODES * CORES_PER_NUMA;

struct PinnedPool {
    pool: ThreadPool,
    threads: usize,
}

struct PoolSets {
    pools_8: Vec<PinnedPool>, // 8 pools × 18 threads
    pools_4: Vec<PinnedPool>, // 4 pools × 36 threads
    pools_2: Vec<PinnedPool>, // 2 pools × 72 threads
    pools_1: Vec<PinnedPool>, // 1 pool  × 144 threads
}

fn build_pinned_pool(cpu_list: Vec<usize>) -> PinnedPool {
    let threads = cpu_list.len();
    let cpu_list_for_handler = cpu_list.clone();

    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |thread_idx| {
            let cpu = cpu_list_for_handler[thread_idx % cpu_list_for_handler.len()];
            let _ = core_affinity::set_for_current(CoreId { id: cpu });
        })
        .build()
        .expect("Failed to build Rayon pool");

    PinnedPool { pool, threads }
}

fn build_option_b_pool_sets() -> PoolSets {
    // 8 pools: split each NUMA node into 2 pools of 18
    let mut pools_8 = Vec::with_capacity(8);
    for numa in 0..NUMA_NODES {
        let base = numa * CORES_PER_NUMA;
        // 2 pools per NUMA node
        for p in 0..2 {
            let start = base + p * (CORES_PER_NUMA / 2); // 18
            let end = start + (CORES_PER_NUMA / 2);
            pools_8.push(build_pinned_pool((start..end).collect()));
        }
    }
    assert_eq!(pools_8.len(), 8);

    // 4 pools: one per NUMA node, 36 threads each
    let mut pools_4 = Vec::with_capacity(4);
    for numa in 0..NUMA_NODES {
        let base = numa * CORES_PER_NUMA;
        pools_4.push(build_pinned_pool((base..(base + CORES_PER_NUMA)).collect()));
    }
    assert_eq!(pools_4.len(), 4);

    // 2 pools: each spans 2 NUMA nodes (72 threads each)
    let pools_2 = vec![
        build_pinned_pool((0..(2 * CORES_PER_NUMA)).collect()),
        build_pinned_pool(((2 * CORES_PER_NUMA)..TOTAL_CORES).collect()),
    ];

    // 1 pool: all cores
    let pools_1 = vec![build_pinned_pool((0..TOTAL_CORES).collect())];

    PoolSets {
        pools_8,
        pools_4,
        pools_2,
        pools_1,
    }
}

/// Choose a pool set to keep ~all cores busy as the tree narrows.
/// Uses largest power-of-two concurrency <= jobs, capped at 8.
fn pick_pools<'a>(sets: &'a PoolSets, jobs: usize) -> &'a [PinnedPool] {
    if jobs >= 8 {
        &sets.pools_8
    } else if jobs >= 4 {
        &sets.pools_4
    } else if jobs >= 2 {
        &sets.pools_2
    } else {
        &sets.pools_1
    }
}

/// Run N jobs with limited concurrency = min(pools.len(), N).
/// Each job runs inside a pinned Rayon pool via `pool.install(|| ...)`,
/// so Halo2’s internal Rayon parallelism stays effective (and localized).
fn map_with_pools<T, E, FJob>(pools: &[PinnedPool], n: usize, job: FJob) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    FJob: Fn(usize) -> Result<T, E> + Sync,
{
    if n == 0 {
        return Ok(vec![]);
    }

    let workers = std::cmp::min(n, pools.len());

    let (tx_job, rx_job) = channel::unbounded::<usize>();
    let (tx_res, rx_res) = channel::unbounded::<(usize, Result<T, E>)>();

    for i in 0..n {
        tx_job.send(i).unwrap();
    }
    drop(tx_job);

    thread::scope(|scope| {
        for pool in pools.iter().take(workers) {
            let rx_job = rx_job.clone();
            let tx_res = tx_res.clone();
            let pool_ref = &pool.pool;
            let job_ref = &job;

            scope.spawn(move || {
                while let Ok(i) = rx_job.recv() {
                    let r = pool_ref.install(|| job_ref(i));
                    let _ = tx_res.send((i, r));
                }
            });
        }
    });

    drop(tx_res);

    let mut out: Vec<Option<Result<T, E>>> = Vec::with_capacity(n);
    out.resize_with(n, || None);
    for _ in 0..n {
        let (i, r) = rx_res.recv().unwrap();
        out[i] = Some(r);
    }

    let mut final_out = Vec::with_capacity(n);
    for item in out {
        match item.expect("missing result") {
            Ok(v) => final_out.push(v),
            Err(e) => return Err(e),
        }
    }
    Ok(final_out)
}

fn io_other(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, msg.into())
}

fn filecoin_srs(k: u32) -> Result<ParamsKZG<Bls12>, std::io::Error> {
    if k > 20 {
        return Err(io_other(format!(
            "No Filecoin SRS available for circuits of size k={}",
            k
        )));
    }

    let srs_dir = env::var("SRS_DIR").unwrap_or_else(|_| "./examples/assets".into());
    let srs_path = format!("{srs_dir}/bls_filecoin_2p{k:?}");
    let mut fetching_path = srs_path.clone();

    if !Path::new(fetching_path.as_str()).exists() {
        fetching_path = format!("{srs_dir}/bls_filecoin_2p20");
    }

    let params_fs = File::open(Path::new(&fetching_path)).map_err(|e| {
        io_other(format!(
            "Failed to open SRS file at '{}': {e}. \
             (Did you download/parse the Filecoin SRS and set SRS_DIR?)",
            fetching_path
        ))
    })?;

    let mut params: ParamsKZG<Bls12> = ParamsKZG::read_custom::<_>(
        &mut BufReader::new(params_fs),
        SerdeFormat::RawBytesUnchecked,
    )
    .map_err(|e| {
        io_other(format!(
            "Failed to read SRS params from '{}': {e}",
            fetching_path
        ))
    })?;

    if fetching_path != srs_path {
        params.downsize(k);
        let mut buf = Vec::new();
        params
            .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
            .map_err(|e| io_other(format!("Failed to serialize downsized params: {e}")))?;

        let mut file = File::create(&srs_path).map_err(|e| {
            io_other(format!(
                "Failed to create SRS cache file '{}': {e}",
                srs_path
            ))
        })?;
        file.write_all(&buf[..]).map_err(|e| {
            io_other(format!(
                "Failed to write downsized SRS params to '{}': {e}",
                srs_path
            ))
        })?;
    }

    Ok(params)
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
    child_vk: VkData,
    child_vk_name: String,
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

#[derive(Clone, Debug)]
pub struct AggCircuit2 {
    child_vk: VkData,
    child_vk_name: String,
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

#[derive(Clone, Debug)]
struct VkData {
    domain: EvaluationDomain<F>,
    cs: ConstraintSystem<F>,
    transcript_repr: F,
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

        let prev_level = scalar_chip.assign(&mut layouter, self.prev_level)?;
        scalar_chip.assert_equal_to_fixed(&mut layouter, &prev_level, self.expected_prev_level)?;
        let next_level = scalar_chip.add_constant(&mut layouter, &prev_level, F::ONE)?;

        let child_vk_val: AssignedNative<F> =
            native_chip.assign_fixed(&mut layouter, self.child_vk.transcript_repr)?;

        let left_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.left_state)?;
        let right_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.right_state)?;
        let next_state =
            poseidon_chip.hash(&mut layouter, &[left_state.clone(), right_state.clone()])?;

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

        let assigned_vk = verifier_chip.assign_vk(
            self.child_vk_name.as_str(),
            &self.child_vk.domain,
            &self.child_vk.cs,
            child_vk_val.clone(),
        )?;

        let (assigned_left_pi, assigned_right_pi) = if self.is_leaf {
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
            let mut left_pi_vec = vec![left_state.clone()];
            left_pi_vec.extend(verifier_chip.as_public_input(&mut layouter, &left_acc)?);
            left_pi_vec.push(prev_level.clone());

            let mut right_pi_vec = vec![right_state.clone()];
            right_pi_vec.extend(verifier_chip.as_public_input(&mut layouter, &right_acc)?);
            right_pi_vec.push(prev_level.clone());

            (left_pi_vec, right_pi_vec)
        };

        let id_point: AssignedForeignPoint<
            midnight_curves::Fq,
            midnight_curves::G1Projective,
            midnight_curves::G1Projective,
        > = curve_chip.assign_fixed(&mut layouter, C::identity())?;

        let mut left_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_vk,
            &[("com_instance", id_point.clone())],
            &[&assigned_left_pi],
            self.left_proof.clone(),
        )?;
        left_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        let mut right_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_vk,
            &[("com_instance", id_point)],
            &[&assigned_right_pi],
            self.right_proof.clone(),
        )?;
        right_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        let mut next_acc = AssignedAccumulator::<S>::accumulate(
            &mut layouter,
            &verifier_chip,
            &scalar_chip,
            &poseidon_chip,
            &[left_proof_acc, left_acc, right_proof_acc, right_acc],
        )?;
        next_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        let next_acc_pi = verifier_chip.as_public_input(&mut layouter, &next_acc)?;

        native_chip.constrain_as_public_input(&mut layouter, &next_state)?;
        for x in next_acc_pi.iter() {
            native_chip.constrain_as_public_input(&mut layouter, x)?;
        }
        native_chip.constrain_as_public_input(&mut layouter, &next_level)?;

        core_decomp_chip.load(&mut layouter)
    }
}

impl Circuit<F> for AggCircuit2 {
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
        let core_decomp_chip = P2RDecompositionChip::new(&config.1, &(K2 as usize - 1));
        let scalar_chip = NativeGadget::new(core_decomp_chip.clone(), native_chip.clone());
        let curve_chip = ForeignEccChip::new(&config.2, &scalar_chip, &scalar_chip);
        let poseidon_chip = PoseidonChip::new(&config.3, &native_chip);
        let verifier_chip = VerifierGadget::new(&curve_chip, &scalar_chip, &poseidon_chip);

        let prev_level = scalar_chip.assign(&mut layouter, self.prev_level)?;
        scalar_chip.assert_equal_to_fixed(&mut layouter, &prev_level, self.expected_prev_level)?;
        let next_level = scalar_chip.add_constant(&mut layouter, &prev_level, F::ONE)?;

        let child_vk_val: AssignedNative<F> =
            native_chip.assign_fixed(&mut layouter, self.child_vk.transcript_repr)?;

        let left_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.left_state)?;
        let right_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.right_state)?;
        let next_state =
            poseidon_chip.hash(&mut layouter, &[left_state.clone(), right_state.clone()])?;

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

        let assigned_vk = verifier_chip.assign_vk(
            self.child_vk_name.as_str(),
            &self.child_vk.domain,
            &self.child_vk.cs,
            child_vk_val.clone(),
        )?;

        let (assigned_left_pi, assigned_right_pi) = if self.is_leaf {
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
            let mut left_pi_vec = vec![left_state.clone()];
            left_pi_vec.extend(verifier_chip.as_public_input(&mut layouter, &left_acc)?);
            left_pi_vec.push(prev_level.clone());

            let mut right_pi_vec = vec![right_state.clone()];
            right_pi_vec.extend(verifier_chip.as_public_input(&mut layouter, &right_acc)?);
            right_pi_vec.push(prev_level.clone());

            (left_pi_vec, right_pi_vec)
        };

        let id_point: AssignedForeignPoint<
            midnight_curves::Fq,
            midnight_curves::G1Projective,
            midnight_curves::G1Projective,
        > = curve_chip.assign_fixed(&mut layouter, C::identity())?;

        let mut left_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_vk,
            &[("com_instance", id_point.clone())],
            &[&assigned_left_pi],
            self.left_proof.clone(),
        )?;
        left_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        let mut right_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_vk,
            &[("com_instance", id_point)],
            &[&assigned_right_pi],
            self.right_proof.clone(),
        )?;
        right_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        let mut next_acc = AssignedAccumulator::<S>::accumulate(
            &mut layouter,
            &verifier_chip,
            &scalar_chip,
            &poseidon_chip,
            &[left_proof_acc, left_acc, right_proof_acc, right_acc],
        )?;
        next_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        let next_acc_pi = verifier_chip.as_public_input(&mut layouter, &next_acc)?;

        native_chip.constrain_as_public_input(&mut layouter, &next_state)?;
        for x in next_acc_pi.iter() {
            native_chip.constrain_as_public_input(&mut layouter, x)?;
        }
        native_chip.constrain_as_public_input(&mut layouter, &next_level)?;

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
    vk: &Vk,
    fixed_bases: &BTreeMap<String, C>,
    proof: &[u8],
    public_inputs: &[F],
) -> Result<Accumulator<S>, std::io::Error> {
    use midnight_proofs::{plonk::prepare, transcript::CircuitTranscript};

    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);

    let dual_msm = prepare::<F, KZGCommitmentScheme<E>, CircuitTranscript<PoseidonState<F>>>(
        vk,
        &[&[C::identity()]],
        &[&[public_inputs]],
        &mut transcript,
    )
    .map_err(|e| io_other(format!("Verification (prepare) failed: {e:?}")))?;

    if !dual_msm.clone().check(&srs.verifier_params()) {
        return Err(io_other("Verification failed: dual MSM did not check"));
    }

    let mut acc: Accumulator<S> = dual_msm.into();
    acc.extract_fixed_bases(fixed_bases);
    acc.collapse();

    if !acc.check(&srs.s_g2().into(), fixed_bases) {
        return Err(io_other(
            "Accumulator failed final check against fixed bases",
        ));
    }

    Ok(acc)
}

fn agg_public_inputs(state: F, acc: &Accumulator<S>, level: F) -> Vec<F> {
    let mut out = Vec::new();
    out.push(state);
    out.extend(AssignedAccumulator::as_public_input(acc));
    out.push(level);
    out
}

fn agg_vk_name_for_level(level: usize) -> String {
    format!("agg_vk_lvl{}", level)
}

#[derive(Clone)]
struct AggLevelKeys {
    level: usize,
    name: String,
    vk: Arc<Vk>,
    pk: Arc<Pk>,
    vk_data: VkData,
    fixed_bases: BTreeMap<String, C>,
}

impl AggLevelKeys {
    fn new(level: usize, name: String, vk: Vk, pk: Pk) -> Self {
        let k = if level == 1 { K } else { K2 };
        let vk_data = VkData {
            domain: EvaluationDomain::new(vk.cs().degree() as u32, k),
            cs: vk.cs().clone(),
            transcript_repr: vk.transcript_repr(),
        };

        let mut fixed_bases = BTreeMap::new();
        fixed_bases.insert(String::from("com_instance"), C::identity());
        fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
            name.as_str(),
            &vk,
        ));

        Self {
            level,
            name,
            vk: Arc::new(vk),
            pk: Arc::new(pk),
            vk_data,
            fixed_bases,
        }
    }
}

struct AggKeyStore {
    levels: Vec<AggLevelKeys>,
}

impl AggKeyStore {
    fn new(levels: Vec<AggLevelKeys>) -> Result<Self, std::io::Error> {
        if levels.is_empty() {
            return Err(io_other("AggKeyStore cannot be empty"));
        }
        let mut seen_names = BTreeSet::new();
        for (i, lvl) in levels.iter().enumerate() {
            let expected_level = i + 1;
            if lvl.level != expected_level {
                return Err(io_other(format!(
                    "AggKeyStore level mismatch at index {}: expected level {}, got {}",
                    i, expected_level, lvl.level
                )));
            }
            if !seen_names.insert(lvl.name.clone()) {
                return Err(io_other(format!(
                    "Duplicate vk_name in AggKeyStore: '{}'",
                    lvl.name
                )));
            }
        }
        Ok(Self { levels })
    }

    fn max_level(&self) -> usize {
        self.levels.len()
    }

    fn get(&self, level: usize) -> Result<&AggLevelKeys, std::io::Error> {
        if level == 0 || level > self.levels.len() {
            return Err(io_other(format!(
                "Requested agg level {} out of range (valid: 1..={})",
                level,
                self.levels.len()
            )));
        }
        Ok(&self.levels[level - 1])
    }
}

fn main() -> Result<(), std::io::Error> {
    // Build adaptive NUMA-pinned pools once and reuse.
    let pool_sets = build_option_b_pool_sets();
    println!("Option B (adaptive) enabled:");
    println!("  8 pools × {} threads/proof", pool_sets.pools_8[0].threads);
    println!("  4 pools × {} threads/proof", pool_sets.pools_4[0].threads);
    println!("  2 pools × {} threads/proof", pool_sets.pools_2[0].threads);
    println!("  1 pool  × {} threads/proof", pool_sets.pools_1[0].threads);

    // -----------------------------
    // Parameters / tree shape
    // -----------------------------
    let num_leaves: usize = 8;
    if !num_leaves.is_power_of_two() {
        return Err(io_other("num_leaves must be a power of two"));
    }
    let max_level: usize = (num_leaves as u32).trailing_zeros() as usize;
    if max_level == 0 {
        return Err(io_other("max_level computed as 0 (unexpected)"));
    }

    // -----------------------------
    // Setup Poseidon circuit
    // -----------------------------
    let poseidon_srs = filecoin_srs(POSEIDON_K)?;
    let poseidon_relation = PoseidonExample;
    let poseidon_vk = compact_std_lib::setup_vk(&poseidon_srs, &poseidon_relation);
    let poseidon_pk = compact_std_lib::setup_pk(&poseidon_relation, &poseidon_vk);
    let poseidon_halo2_vk = poseidon_vk.vk();
    let poseidon_vk_data = VkData {
        domain: EvaluationDomain::new(poseidon_halo2_vk.cs().degree() as u32, POSEIDON_K),
        cs: poseidon_halo2_vk.cs().clone(),
        transcript_repr: poseidon_halo2_vk.transcript_repr(),
    };

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
    let agg_srs1 = filecoin_srs(K)?;
    let agg_srs2 = filecoin_srs(K2)?;

    assert_eq!(
        poseidon_srs.s_g2(),
        agg_srs2.s_g2(),
        "poseidon vs agg_srs2 s_g2 mismatch"
    );
    assert_eq!(
        agg_srs1.s_g2(),
        agg_srs2.s_g2(),
        "agg_srs1 vs agg_srs2 s_g2 mismatch"
    );

    // -----------------------------
    // Precompute all AGG vk names
    // -----------------------------
    let agg_vk_names: Vec<String> = (1..=max_level).map(agg_vk_name_for_level).collect();

    // -----------------------------
    // Build combined fixed base name list
    // -----------------------------
    let combined_fixed_base_names: Vec<String> = {
        let mut set = BTreeSet::new();
        let mut out = Vec::new();

        for name in fixed_base_names_for("poseidon_vk", &poseidon_vk_data.cs) {
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
    // Keygen one AGG VK/PK per level
    // -----------------------------
    let mut agg_levels: Vec<AggLevelKeys> = Vec::with_capacity(max_level);

    for level in 1..=max_level {
        let (child_vk, child_vk_name, expected_prev_level, is_leaf) = if level == 1 {
            (
                poseidon_vk_data.clone(),
                "poseidon_vk".to_string(),
                F::ZERO,
                true,
            )
        } else {
            let child_level = level - 1;
            let child = agg_levels
                .get(child_level - 1)
                .ok_or_else(|| {
                    io_other(format!("Missing child level {} during keygen", child_level))
                })?
                .vk_data
                .clone();
            let child_name = agg_vk_names[child_level - 1].clone();
            (child, child_name, F::from((level as u64) - 1), false)
        };

        let default_circuit1 = AggCircuit {
            child_vk: child_vk.clone(),
            child_vk_name: child_vk_name.clone(),
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
        let default_circuit2 = AggCircuit2 {
            child_vk,
            child_vk_name,
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
        let k = if level == 1 { K } else { K2 };
        let vk: VerifyingKey<midnight_curves::Fq, KZGCommitmentScheme<Bls12>> = if level == 1 {
            keygen_vk_with_k(&agg_srs1, &default_circuit1, k).map_err(|e| {
                io_other(format!("keygen_vk_with_k failed at level {}: {e:?}", level))
            })?
        } else {
            keygen_vk_with_k(&agg_srs2, &default_circuit2, k).map_err(|e| {
                io_other(format!("keygen_vk_with_k failed at level {}: {e:?}", level))
            })?
        };
        let pk: ProvingKey<midnight_curves::Fq, KZGCommitmentScheme<Bls12>> = if level == 1 {
            keygen_pk(vk.clone(), &default_circuit1)
                .map_err(|e| io_other(format!("keygen_pk failed at level {}: {e:?}", level)))?
        } else {
            keygen_pk(vk.clone(), &default_circuit2)
                .map_err(|e| io_other(format!("keygen_pk failed at level {}: {e:?}", level)))?
        };

        let name = agg_vk_names[level - 1].clone();
        println!("Computed {} vk/pk in {:?}", name, start.elapsed());

        agg_levels.push(AggLevelKeys::new(level, name, vk, pk));
    }

    let agg_store = AggKeyStore::new(agg_levels)?;

    // -----------------------------
    // Build combined fixed bases map
    // -----------------------------
    let mut combined_fixed_bases = BTreeMap::new();
    for (name, base) in poseidon_fixed_bases.iter() {
        combined_fixed_bases.insert(name.clone(), *base);
    }
    for level in 1..=agg_store.max_level() {
        for (name, base) in agg_store.get(level)?.fixed_bases.iter() {
            combined_fixed_bases.insert(name.clone(), *base);
        }
    }

    // -----------------------------
    // Build "global trivial" accumulator
    // -----------------------------
    let trivial_poseidon =
        trivial_acc_with_names(&fixed_base_names_for("poseidon_vk", &poseidon_vk_data.cs));

    let mut trivial_all: Vec<Accumulator<S>> = vec![trivial_poseidon];
    for level in 1..=agg_store.max_level() {
        let vk_name = agg_store.get(level)?.name.as_str();
        let cs = agg_store.get(level)?.vk.cs();
        let t = trivial_acc_with_names(&fixed_base_names_for(vk_name, cs));
        trivial_all.push(t);
    }

    let mut trivial_combined = Accumulator::accumulate(&trivial_all);
    trivial_combined.collapse();

    // -----------------------------
    // Generate Poseidon leaf proofs (NOW parallelized with adaptive pools)
    // -----------------------------
    println!("Creating {} POSEIDON leaf proofs...", num_leaves);
    let poseidon_pools = pick_pools(&pool_sets, num_leaves);
    println!(
        "Poseidon stage: {} jobs, using {} pools × {} threads/proof",
        num_leaves,
        poseidon_pools.len(),
        poseidon_pools[0].threads
    );

    let poseidon_proofs: Vec<(F, [F; 3], Vec<u8>)> =
        map_with_pools(poseidon_pools, num_leaves, |i| {
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
                .map_err(|e| io_other(format!("Poseidon proof failed for leaf {i}: {e:?}")))?;
                transcript.finalize()
            };

            Ok::<_, std::io::Error>((instance, witness, proof))
        })?;

    // -----------------------------
    // Create leaf aggregation layer (AGG level 1)
    // -----------------------------
    println!("\nCreating {} leaf AGG nodes...", num_leaves / 2);

    let leaf_level = 1usize;
    let leaf_keys = agg_store.get(leaf_level)?;
    let leaf_agg_vk_name = leaf_keys.name.clone();

    let leaf_jobs = num_leaves / 2;
    let leaf_pools = pick_pools(&pool_sets, leaf_jobs);
    println!(
        "Leaf AGG stage: {} jobs, using {} pools × {} threads/proof",
        leaf_jobs,
        leaf_pools.len(),
        leaf_pools[0].threads
    );

    let mut current_level: Vec<TreeNode> = map_with_pools(leaf_pools, leaf_jobs, |i| {
        use midnight_circuits::instructions::hash::HashCPU;

        let (left_state, _, left_proof) = &poseidon_proofs[i * 2];
        let (right_state, _, right_proof) = &poseidon_proofs[i * 2 + 1];

        let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[*left_state, *right_state]);

        let circuit = AggCircuit {
            child_vk: poseidon_vk_data.clone(),
            child_vk_name: "poseidon_vk".to_string(),
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
        )?;
        let proof_acc_right = verify_and_extract_acc(
            &poseidon_srs,
            poseidon_vk.vk(),
            &poseidon_fixed_bases,
            right_proof,
            &[*right_state],
        )?;

        let mut accumulated_pi = Accumulator::accumulate(&[
            proof_acc_left,
            trivial_combined.clone(),
            proof_acc_right,
            trivial_combined.clone(),
        ]);
        accumulated_pi.collapse();

        let public_inputs = agg_public_inputs(state, &accumulated_pi, F::ONE);

        let start = Instant::now();
        let proof = {
            let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
            create_proof::<
                F,
                KZGCommitmentScheme<E>,
                CircuitTranscript<PoseidonState<F>>,
                AggCircuit,
            >(
                &agg_srs1,
                leaf_keys.pk.as_ref(),
                &[circuit],
                1,
                &[&[&[], &public_inputs]],
                OsRng,
                &mut transcript,
            )
            .map_err(|e| io_other(format!("Leaf AGG proof failed for node {i}: {e:?}")))?;
            transcript.finalize()
        };
        println!(
            "Leaf AGG {} ({}) created in {:?}",
            i,
            leaf_agg_vk_name,
            start.elapsed()
        );

        if !accumulated_pi.check(&agg_srs2.s_g2().into(), &combined_fixed_bases) {
            return Err(io_other(format!(
                "Leaf node {i}: accumulated PI accumulator did not check against combined fixed bases"
            )));
        }

        let proof_acc = verify_and_extract_acc(
            &agg_srs1,
            leaf_keys.vk.as_ref(),
            &leaf_keys.fixed_bases,
            &proof,
            &public_inputs,
        )?;

        Ok::<_, std::io::Error>(TreeNode {
            state,
            proof,
            proof_acc,
            pi_acc: accumulated_pi,
        })
    })?;

    // -----------------------------
    // Build internal layers
    // -----------------------------
    let mut child_level: usize = 1;
    while current_level.len() > 1 {
        let parent_level = child_level + 1;
        let parent_keys = agg_store.get(parent_level)?;
        let parent_vk_name = parent_keys.name.clone();

        let jobs = current_level.len() / 2;
        let pools = pick_pools(&pool_sets, jobs);
        println!(
            "\nBuilding AGG level {} ({}) with {} nodes... using {} pools × {} threads/proof",
            parent_level,
            parent_vk_name,
            jobs,
            pools.len(),
            pools[0].threads
        );

        let child_keys = agg_store.get(child_level)?;
        let child_vk_data = child_keys.vk_data.clone();
        let child_vk_name = child_keys.name.clone();

        let next_level: Vec<TreeNode> = map_with_pools(pools, jobs, |i| {
            use midnight_circuits::instructions::hash::HashCPU;

            let left = &current_level[i * 2];
            let right = &current_level[i * 2 + 1];

            let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

            let circuit = AggCircuit2 {
                child_vk: child_vk_data.clone(),
                child_vk_name: child_vk_name.clone(),
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

            let public_inputs =
                agg_public_inputs(state, &accumulated_pi, F::from(parent_level as u64));

            let start = Instant::now();
            let proof = {
                let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
                create_proof::<
                    F,
                    KZGCommitmentScheme<E>,
                    CircuitTranscript<PoseidonState<F>>,
                    AggCircuit2,
                >(
                    &agg_srs2,
                    parent_keys.pk.as_ref(),
                    &[circuit],
                    1,
                    &[&[&[], &public_inputs]],
                    OsRng,
                    &mut transcript,
                )
                .map_err(|e| {
                    io_other(format!(
                        "Internal AGG proof failed at level {parent_level}, node {i}: {e:?}"
                    ))
                })?;
                transcript.finalize()
            };
            println!(
                "Level {} node {} ({}) created in {:?}",
                parent_level,
                i,
                parent_vk_name,
                start.elapsed()
            );

            if !accumulated_pi.check(&agg_srs2.s_g2().into(), &combined_fixed_bases) {
                return Err(io_other(format!(
                    "Level {parent_level} node {i}: accumulated PI accumulator did not check against combined fixed bases"
                )));
            }

            let proof_acc = verify_and_extract_acc(
                &agg_srs2,
                parent_keys.vk.as_ref(),
                &parent_keys.fixed_bases,
                &proof,
                &public_inputs,
            )?;

            Ok::<_, std::io::Error>(TreeNode {
                state,
                proof,
                proof_acc,
                pi_acc: accumulated_pi,
            })
        })?;

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
    if root.state != expected_root {
        return Err(io_other("Root state mismatch"));
    }
    println!("✓ Root verification successful!");

    Ok(())
}
