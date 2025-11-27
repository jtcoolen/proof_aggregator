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
    verifier::{
        self, Accumulator, AssignedAccumulator, AssignedVk, BlstrsEmulation, SelfEmulation,
        VerifierGadget,
    },
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
use rayon::prelude::*;
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

// POSEIDON relation
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

// AGG Circuit
#[derive(Clone, Debug)]
pub struct AggCircuit {
    agg_vk: (EvaluationDomain<F>, ConstraintSystem<F>, Value<F>),
    // Name of the VK this circuit is *verifying* (e.g. "leaf_agg_vk" at level 1, "agg_vk" at higher levels)
    agg_vk_name: &'static str,
    // If present, this circuit is in "leaf" mode and verifies Poseidon proofs directly
    poseidon_vk: Option<(EvaluationDomain<F>, ConstraintSystem<F>, Value<F>)>,
    left_state: Value<F>,
    right_state: Value<F>,
    left_proof: Value<Vec<u8>>,
    right_proof: Value<Vec<u8>>,
    left_acc: Value<Accumulator<S>>,
    right_acc: Value<Accumulator<S>>,

    // Concrete fixed-base names for the RHS accumulator MSMs
    fixed_base_names: Vec<String>,

    // NEW: When verifying leaf-agg proofs (level 1), the child PI must start with Poseidon VK encoding.
    // Carry those field elements here as witnesses for the child PI prefix.
    child_poseidon_vk_pi: Option<Vec<F>>,
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

        // Determine which VK is used for verification:
        // - poseidon_vk for leaf AGG circuits (verifying Poseidon proofs)
        // - agg_vk (with agg_vk_name) for internal levels (verifying leaf-agg or agg proofs)
        let (vk_domain, vk_cs, vk_value, vk_name) = if let Some((d, cs, v)) = &self.poseidon_vk {
            (d, cs, v, "poseidon_vk")
        } else {
            (&self.agg_vk.0, &self.agg_vk.1, &self.agg_vk.2, self.agg_vk_name)
        };

        let assigned_vk: AssignedVk<S> = verifier_chip.assign_vk_as_public_input(
            &mut layouter,
            vk_name,
            vk_domain,
            vk_cs,
            *vk_value,
        )?;

        let left_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.left_state)?;
        let right_state: AssignedNative<F> = scalar_chip.assign(&mut layouter, self.right_state)?;

        // Compute next_state = poseidon([left_state, right_state])
        let next_state =
            poseidon_chip.hash(&mut layouter, &[left_state.clone(), right_state.clone()])?;
        scalar_chip.constrain_as_public_input(&mut layouter, &next_state)?;

        let id_point: AssignedForeignPoint<F, C, _> =
            curve_chip.assign_fixed(&mut layouter, C::identity())?;

        // Use the precomputed fixed-base names (ensures consistency with the off-circuit accumulators)
        let fixed_base_names = self.fixed_base_names.clone();

        // Left child
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

        // For leaf nodes (verifying Poseidon proofs), the child public input is just the state.
        // For internal nodes (verifying AGG/leaf-AGG proofs), we keep [vk, state, acc] as before.
        let is_leaf = self.poseidon_vk.is_some();

        let assigned_left_pi = if is_leaf {
            vec![left_state.clone()]
        } else {
            let mut v: Vec<AssignedNative<F>> = Vec::new();
            if self.agg_vk_name == "leaf_agg_vk" {
                // Level 1: child is a leaf-agg proof → child PI starts with Poseidon VK encoding
                let vk_pi = self
                    .child_poseidon_vk_pi
                    .as_ref()
                    .expect("missing child_poseidon_vk_pi for level-1");
                let mut assigned_prefix = vk_pi
                    .iter()
                    .map(|&s| scalar_chip.assign(&mut layouter, Value::known(s)))
                    .collect::<Result<Vec<_>, Error>>()?;
                v.append(&mut assigned_prefix);
            } else {
                // Higher levels: child is an AGG proof → child PI starts with AGG VK encoding
                v.extend(verifier_chip.as_public_input(&mut layouter, &assigned_vk)?);
            }
            v.push(left_state.clone());
            v.extend(verifier_chip.as_public_input(&mut layouter, &left_acc)?);
            v
        };

        let mut left_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_vk,
            &[("com_instance", id_point.clone())],
            &[&assigned_left_pi],
            self.left_proof.clone(),
        )?;
        left_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        // Right child
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

        let assigned_right_pi = if is_leaf {
            vec![right_state.clone()]
        } else {
            let mut v: Vec<AssignedNative<F>> = Vec::new();
            if self.agg_vk_name == "leaf_agg_vk" {
                let vk_pi = self
                    .child_poseidon_vk_pi
                    .as_ref()
                    .expect("missing child_poseidon_vk_pi for level-1");
                let mut assigned_prefix = vk_pi
                    .iter()
                    .map(|&s| scalar_chip.assign(&mut layouter, Value::known(s)))
                    .collect::<Result<Vec<_>, Error>>()?;
                v.append(&mut assigned_prefix);
            } else {
                v.extend(verifier_chip.as_public_input(&mut layouter, &assigned_vk)?);
            }
            v.push(right_state.clone());
            v.extend(verifier_chip.as_public_input(&mut layouter, &right_acc)?);
            v
        };

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
        verifier_chip.constrain_as_public_input(&mut layouter, &next_acc)?;

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

fn fixed_base_names_for(
    vk_name: &str,
    cs: &midnight_proofs::plonk::ConstraintSystem<F>,
) -> Vec<String> {
    let mut names = vec![String::from("com_instance"), String::from("~G")];
    names.extend(midnight_circuits::verifier::fixed_base_names::<S>(
        vk_name,
        cs.num_fixed_columns() + cs.num_selectors(),
        cs.permutation().columns.len(),
    ));
    names
}

fn trivial_acc_with_names(names: &[String]) -> midnight_circuits::verifier::Accumulator<S> {
    use midnight_circuits::verifier::Msm;
    use std::collections::BTreeMap;
    let fixed: BTreeMap<String, F> = names.iter().cloned().map(|n| (n, F::ZERO)).collect();

    midnight_circuits::verifier::Accumulator::<S>::new(
        Msm::new(&[C::default()], &[F::ONE], &BTreeMap::new()),
        Msm::new(&[C::default()], &[F::ONE], &fixed),
    )
}

fn main() {
    // Setup POSEIDON circuit
    let poseidon_srs = filecoin_srs(POSEIDON_K);
    let poseidon_relation = PoseidonExample;
    let poseidon_vk = compact_std_lib::setup_vk(&poseidon_srs, &poseidon_relation);
    let poseidon_pk = compact_std_lib::setup_pk(&poseidon_relation, &poseidon_vk);

    // Extract VK for AGG circuit to use
    let poseidon_halo2_vk: &VerifyingKey<F, KZGCommitmentScheme<E>> = poseidon_vk.vk();
    let poseidon_vk_data = (
        EvaluationDomain::new(poseidon_vk.vk().cs().degree() as u32, POSEIDON_K),
        poseidon_vk.vk().cs().clone(),
        Value::known(poseidon_halo2_vk.transcript_repr()),
    );

    // Compute fixed bases for POSEIDON VK
    let mut poseidon_fixed_bases = BTreeMap::new();
    poseidon_fixed_bases.insert(String::from("com_instance"), C::identity());
    poseidon_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "poseidon_vk",
        poseidon_halo2_vk,
    ));

    // Setup AGG circuit (CS only, used to derive shapes for keygen)
    let agg_k = K;
    let mut agg_cs = ConstraintSystem::default();
    configure_agg_circuit(&mut agg_cs);
    let agg_domain = EvaluationDomain::new(agg_cs.degree() as u32, agg_k);

    // Fixed-base names for keygen circuits (using CS only)
    let combined_fixed_base_names_keygen: Vec<String> = {
        let poseidon_fb = fixed_base_names_for("poseidon_vk", &poseidon_vk_data.1);
        let leaf_agg_fb = fixed_base_names_for("leaf_agg_vk", &agg_cs);
        let agg_fb = fixed_base_names_for("agg_vk", &agg_cs);

        let mut set = BTreeSet::new();
        let mut v = Vec::new();
        for name in poseidon_fb
            .iter()
            .chain(leaf_agg_fb.iter())
            .chain(agg_fb.iter())
        {
            if set.insert(name.clone()) {
                v.push(name.clone());
            }
        }
        v
    };

    // Default AGG circuit for keygen (unknown witnesses) – this is the "agg_vk" relation
    let default_agg_circuit = AggCircuit {
        agg_vk: (agg_domain.clone(), agg_cs.clone(), Value::unknown()),
        agg_vk_name: "agg_vk",
        poseidon_vk: None,
        left_state: Value::unknown(),
        right_state: Value::unknown(),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_acc: Value::unknown(),
        right_acc: Value::unknown(),
        fixed_base_names: combined_fixed_base_names_keygen.clone(),
        child_poseidon_vk_pi: None,
    };

    let agg_srs = filecoin_srs(agg_k);
    let start = Instant::now();
    let agg_vk = keygen_vk_with_k(&agg_srs, &default_agg_circuit, agg_k).unwrap();
    let agg_pk = keygen_pk(agg_vk.clone(), &default_agg_circuit).unwrap();
    println!("Computed AGG vk and pk in {:?}", start.elapsed());

    // Default leaf-agg circuit for keygen (same CS as agg pre-keygen)
    // This circuit verifies POSEIDON proofs (poseidon_vk is Some)
    let default_leaf_agg_circuit = AggCircuit {
        agg_vk: (agg_domain.clone(), agg_cs.clone(), Value::unknown()),
        agg_vk_name: "leaf_agg_vk", // name used when this vk is later used at level 1
        poseidon_vk: Some(poseidon_vk_data.clone()),
        left_state: Value::unknown(),
        right_state: Value::unknown(),
        left_proof: Value::unknown(),
        right_proof: Value::unknown(),
        left_acc: Value::unknown(),
        right_acc: Value::unknown(),
        fixed_base_names: combined_fixed_base_names_keygen.clone(),
        child_poseidon_vk_pi: None,
    };

    let leaf_agg_vk: VerifyingKey<F, KZGCommitmentScheme<E>> =
        keygen_vk_with_k(&agg_srs, &default_leaf_agg_circuit, agg_k).unwrap();
    let leaf_agg_pk = keygen_pk(leaf_agg_vk.clone(), &default_leaf_agg_circuit).unwrap();
    println!("Computed leaf AGG vk and pk in {:?}", start.elapsed());

    // Fixed bases per VK (consistent prefixes)
    let mut agg_fixed_bases = BTreeMap::new();
    agg_fixed_bases.insert(String::from("com_instance"), C::identity());
    agg_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "agg_vk",
        &agg_vk,
    ));

    let mut leaf_agg_fixed_bases: BTreeMap<String, midnight_curves::G1Projective> = BTreeMap::new();
    leaf_agg_fixed_bases.insert(String::from("com_instance"), C::identity());
    leaf_agg_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "leaf_agg_vk",
        &leaf_agg_vk,
    ));

    // Combined bases (optional diagnostics)
    let mut combined_fixed_bases: BTreeMap<String, midnight_curves::G1Projective> = BTreeMap::new();
    combined_fixed_bases.insert(String::from("com_instance"), C::identity());
    combined_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "agg_vk",
        &agg_vk,
    ));
    combined_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "leaf_agg_vk",
        &leaf_agg_vk,
    ));
    combined_fixed_bases.extend(midnight_circuits::verifier::fixed_bases::<S>(
        "poseidon_vk",
        poseidon_vk.vk(),
    ));

    let agg_srs = Arc::new(agg_srs);
    let agg_vk = Arc::new(agg_vk);
    let agg_pk = Arc::new(agg_pk);

    // Names for each VK shape (consistent prefixes; include "~G")
    let poseidon_fixed_base_names = fixed_base_names_for("poseidon_vk", &poseidon_vk_data.1);
    let leaf_agg_fixed_base_names = fixed_base_names_for("leaf_agg_vk", &leaf_agg_vk.cs());
    let agg_fixed_base_names = fixed_base_names_for("agg_vk", &agg_vk.cs());

    let combined_fixed_base_names: Vec<String> = {
        let mut set = BTreeSet::new();
        let mut v = Vec::new();
        for name in poseidon_fixed_base_names
            .iter()
            .chain(leaf_agg_fixed_base_names.iter())
            .chain(agg_fixed_base_names.iter())
        {
            if set.insert(name.clone()) {
                v.push(name.clone());
            }
        }
        v
    };

    // Trivial accumulators *with matching shapes*
    let trivial_poseidon_pi: Accumulator<S> = trivial_acc_with_names(&poseidon_fixed_base_names);

    let agg_fixed_bases = Arc::new(agg_fixed_bases);
    let leaf_agg_fixed_bases = Arc::new(leaf_agg_fixed_bases);
    let poseidon_fixed_bases = Arc::new(poseidon_fixed_bases);
    let trivial_leaf_agg: Accumulator<S> = trivial_acc_with_names(&leaf_agg_fixed_base_names);
    let trivial_agg: Accumulator<S> = trivial_acc_with_names(&agg_fixed_base_names);

    let mut trivial_combined =
        Accumulator::accumulate(&[trivial_poseidon_pi, trivial_leaf_agg, trivial_agg]);
    trivial_combined.collapse(); // keep (1,1) for committed sides
    println!(
        "trivial acc combined {:?}",
        trivial_combined.rhs().fixed_base_scalars()
    );

    // Create POSEIDON leaf proofs
    let num_leaves = 4;
    println!("Creating {} POSEIDON leaf proofs...", num_leaves);

    let poseidon_proofs: Vec<(F, [F; 3], Vec<u8>)> = (0..num_leaves)
        .into_iter()
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
                .expect("Leaf AGG proof failed");
                transcript.finalize()
            };

            println!("POSEIDON leaf {} created", i);
            (instance, witness, proof)
        })
        .collect();

    // Create leaf AGG nodes that verify POSEIDON proofs
    println!("\nCreating {} leaf AGG nodes...", num_leaves / 2);

    let mut current_level: Vec<TreeNode> = (0..num_leaves / 2)
        .into_iter()
        .map(|i| {
            let (left_state, _, left_proof) = &poseidon_proofs[i * 2];
            let (right_state, _, right_proof) = &poseidon_proofs[i * 2 + 1];

            // Compute expected state: poseidon([left_state, right_state])
            use midnight_circuits::instructions::hash::HashCPU;
            let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[*left_state, *right_state]);

            let leaf_agg_vk_data = (
                EvaluationDomain::<F>::new(leaf_agg_vk.cs().degree() as u32, K),
                leaf_agg_vk.cs().clone(),
                Value::known(leaf_agg_vk.transcript_repr()),
            );
            let circuit = AggCircuit {
                // This circuit is a "leaf agg"; it verifies POSEIDON proofs
                agg_vk: leaf_agg_vk_data, // unused; poseidon_vk below selects leaf mode
                agg_vk_name: "leaf_agg_vk",
                poseidon_vk: Some(poseidon_vk_data.clone()),
                left_state: Value::known(*left_state),
                right_state: Value::known(*right_state),
                left_proof: Value::known(left_proof.clone()),
                right_proof: Value::known(right_proof.clone()),
                left_acc: Value::known(trivial_combined.clone()),
                right_acc: Value::known(trivial_combined.clone()),
                fixed_base_names: combined_fixed_base_names.clone(),
                child_poseidon_vk_pi: None,
            };

            let proof_acc_left = verify_and_extract_acc(
                &poseidon_srs,
                &poseidon_vk.vk(),
                &poseidon_fixed_bases,
                &left_proof,
                &[*left_state],
            );

            let proof_acc_right = verify_and_extract_acc(
                &poseidon_srs,
                &poseidon_vk.vk(),
                &poseidon_fixed_bases,
                &right_proof,
                &[*right_state],
            );

            let mut accumulated_pi = Accumulator::accumulate(&[
                proof_acc_left.clone(),
                trivial_combined.clone(),
                proof_acc_right.clone(),
                trivial_combined.clone(),
            ]);
            accumulated_pi.collapse();

            // Public input for leaf agg: [poseidon_vk, state, accumulated_pi]
            let mut public_inputs = AssignedVk::<S>::as_public_input(&poseidon_vk.vk());
            public_inputs.extend(AssignedNative::<F>::as_public_input(&state));
            public_inputs.extend(AssignedAccumulator::as_public_input(&accumulated_pi));

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

            // Verify leaf-agg proof with LEAF_AGG fixed bases
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
                pi_acc: accumulated_pi.clone(),
            }
        })
        .collect();

    // Build internal AGG tree
    let mut level = 0;
    while current_level.len() > 1 {
        level += 1;
        println!(
            "\nBuilding AGG level {} with {} nodes...",
            level,
            current_level.len() / 2
        );

        let next_level: Vec<TreeNode> = (0..current_level.len() / 2)
            .into_iter()
            .map(|pair_idx| {
                let i = pair_idx * 2;
                let left = current_level[i].clone();
                let right = current_level[i + 1].clone();

                use midnight_circuits::instructions::hash::HashCPU;
                let state = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[left.state, right.state]);

                let leaf_agg_vk_data = (
                    EvaluationDomain::new(leaf_agg_vk.cs().degree() as u32, K),
                    leaf_agg_vk.cs().clone(),
                    Value::known(leaf_agg_vk.transcript_repr()),
                );

                // At level 1 we verify leaf-agg proofs (vk = leaf_agg_vk)
                // At higher levels we verify agg proofs (vk = agg_vk)
                let circuit_agg_vk = if level == 1 {
                    leaf_agg_vk_data
                } else {
                    (
                        agg_domain.clone(),
                        agg_cs.clone(),
                        Value::known(agg_vk.transcript_repr()),
                    )
                };
                let circuit = AggCircuit {
                    agg_vk: circuit_agg_vk,
                    agg_vk_name: if level == 1 { "leaf_agg_vk" } else { "agg_vk" },
                    poseidon_vk: None,
                    left_state: Value::known(left.state),
                    right_state: Value::known(right.state),
                    left_proof: Value::known(left.proof.clone()),
                    right_proof: Value::known(right.proof.clone()),
                    left_acc: Value::known(left.pi_acc.clone()),
                    right_acc: Value::known(right.pi_acc.clone()),
                    fixed_base_names: combined_fixed_base_names.clone(),
                    // Level 1: child PI must start with Poseidon VK encoding
                    child_poseidon_vk_pi: if level == 1 {
                        Some(AssignedVk::<S>::as_public_input(poseidon_vk.vk()))
                    } else {
                        None
                    },
                };

                let mut accumulated_pi = Accumulator::accumulate(&[
                    left.proof_acc.clone(),
                    left.pi_acc.clone(),
                    right.proof_acc.clone(),
                    right.pi_acc.clone(),
                ]);
                accumulated_pi.collapse();

                // Public inputs depend on which relation this level verifies
                let input_agg_vk = if level == 1 {
                    &leaf_agg_vk
                } else {
                    agg_vk.as_ref()
                };

                let input_agg_pk = if level == 1 {
                    &leaf_agg_pk
                } else {
                    agg_pk.as_ref()
                };

                let mut public_inputs = AssignedVk::<S>::as_public_input(input_agg_vk);
                public_inputs.extend(AssignedNative::<F>::as_public_input(&state));
                public_inputs.extend(AssignedAccumulator::as_public_input(&accumulated_pi));

                println!("about to produce an internal AGG proof");
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
                        &input_agg_pk,
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
                    pair_idx,
                    start.elapsed()
                );

                // Verify internal AGG proof:
                //  - level 1: vk = leaf_agg_vk, bases = leaf_agg_fixed_bases
                //  - level >1: vk = agg_vk, bases = agg_fixed_bases
                let (verify_vk, verify_fixed_bases) = if level == 1 {
                    (&leaf_agg_vk, &*leaf_agg_fixed_bases)
                } else {
                    (agg_vk.as_ref(), &*agg_fixed_bases)
                };

                let proof_acc = verify_and_extract_acc(
                    &agg_srs,
                    verify_vk,
                    verify_fixed_bases,
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

    // Verify by recomputing from POSEIDON inputs
    use midnight_circuits::instructions::hash::HashCPU;
    let leaf_states: Vec<F> = poseidon_proofs.iter().map(|(s, _, _)| *s).collect();
    let level0_0 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[leaf_states[0], leaf_states[1]]);
    let level0_1 = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[leaf_states[2], leaf_states[3]]);
    let expected_root = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[level0_0, level0_1]);

    println!(
        "Expected root (recomputed from POSEIDON proofs): {:?}",
        expected_root
    );
    assert_eq!(root.state, expected_root, "Root state mismatch!");
    println!("✓ Root verification successful!");
}

fn verify_and_extract_acc(
    srs: &ParamsKZG<Bls12>,
    vk: &midnight_proofs::plonk::VerifyingKey<F, KZGCommitmentScheme<E>>,
    fixed_bases: &BTreeMap<String, C>,
    proof: &[u8],
    plain_public_inputs: &[F],
) -> Accumulator<S> {
    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(proof);
    let committed_bases: &[&[C]] = &[&[C::identity()]];
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
