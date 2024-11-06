NUM_GEN_SAMPLES=1000

mkdir -p testing_output/corgan
python -m synthEHRella.run_generation \
    corgan --real_training_data_path testing_output/processed_mimic3.matrix \
    --ckpt_dir testing_output/corgan \
    --num_gen_samples $NUM_GEN_SAMPLES --params "--n_epochs 1000"

mkdir -p testing_output/medgan
python -m synthEHRella.run_generation \
    medgan --real_training_data_path testing_output/processed_mimic3.matrix \
    --ckpt_dir testing_output/medgan \
    --num_gen_samples $NUM_GEN_SAMPLES --params "--n_epochs 1000"

mkdir -p testing_output/vae
python -m synthEHRella.run_generation \
    vae --real_training_data_path testing_output/processed_mimic3.matrix \
    --ckpt_dir testing_output/vae \
    --num_gen_samples $NUM_GEN_SAMPLES --params "--epochs 1000"

mkdir -p testing_output/promptehr
python -m synthEHRella.run_generation \
    promptehr --real_training_data_path testing_output/processed_mimic3.matrix \
    --ckpt_dir testing_output/promptehr \
    --num_gen_samples $NUM_GEN_SAMPLES

mkdir -p testing_output/ehrdiff
python -m synthEHRella.run_generation \
    ehrdiff --real_training_data_path tests/X_num_train.npy \
    --ckpt_dir testing_output/ehrdiff \
    --num_gen_samples $NUM_GEN_SAMPLES --params "train.n_epochs=50"

module load openjdk/18.0.1.1
mkdir -p testing_output/synthea
python -m synthEHRella.run_generation synthea \
    --ckpt_dir testing_output/synthea \
    --num_gen_samples $NUM_GEN_SAMPLES

module load R/4.2.0
mkdir -p testing_output/plasmode
python -m synthEHRella.run_generation \
    plasmode --real_training_data_path preprocessed_mimiciii_for_plasmode_with_demo_survival.csv \
    --ckpt_dir testing_output/plasmode \
    --num_gen_samples $NUM_GEN_SAMPLES


mkdir -p testing_output/resample
python -m synthEHRella.run_generation resample \
    --real_training_data_path testing_output/processed_mimic3.matrix \
    --ckpt_dir testing_output/resample \
    --num_gen_samples 1000

mkdir -p testing_output/pbr
python -m synthEHRella.run_generation pbr \
    --real_training_data_path testing_output/processed_mimic3.matrix \
    --ckpt_dir testing_output/pbr \
    --num_gen_samples 1000