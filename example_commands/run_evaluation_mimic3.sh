python -m synthEHRella.run_evaluation corgan \
    --synthetic_data_path testing_output/postprocessed/corgan-synthetic-phecodexm.npy \
    --real_eval_data_path testing_output/mimic3-real-phecodexm.npy \
    --output_dir testing_output/evaluation/corgan-mimic3-evaluation.json

python -m synthEHRella.run_evaluation ehrdiff \
    --synthetic_data_path testing_output/postprocessed/ehrdiff-synthetic-phecodexm.npy \
    --real_eval_data_path testing_output/mimic3-real-phecodexm.npy \
    --output_dir testing_output/evaluation/ehrdiff-mimic3-evaluation.json

python -m synthEHRella.run_evaluation medgan \
    --synthetic_data_path testing_output/postprocessed/medgan-synthetic-phecodexm.npy \
    --real_eval_data_path testing_output/mimic3-real-phecodexm.npy \
    --output_dir testing_output/evaluation/medgan-mimic3-evaluation.json

python -m synthEHRella.run_evaluation plasmode \
    --synthetic_data_path testing_output/postprocessed/plasmode-synthetic-phecodexm.npy \
    --real_eval_data_path testing_output/mimic3-real-phecodexm.npy \
    --output_dir testing_output/evaluation/plasmode-mimic3-evaluation.json

python -m synthEHRella.run_evaluation promptehr \
    --synthetic_data_path testing_output/postprocessed/promptehr-synthetic-phecodexm.npy \
    --real_eval_data_path testing_output/mimic3-real-phecodexm.npy \
    --output_dir testing_output/evaluation/promptehr-mimic3-evaluation.json

python -m synthEHRella.run_evaluation synthea \
    --synthetic_data_path testing_output/postprocessed/synthea-synthetic-phecodexm.npy \
    --real_eval_data_path testing_output/mimic3-real-phecodexm.npy \
    --output_dir testing_output/evaluation/synthea-mimic3-evaluation.json

python -m synthEHRella.run_evaluation vae \
    --synthetic_data_path testing_output/postprocessed/vae-synthetic-phecodexm.npy \
    --real_eval_data_path testing_output/mimic3-real-phecodexm.npy \
    --output_dir testing_output/evaluation/vae-mimic3-evaluation.json

python -m synthEHRella.run_evaluation resample \
    --synthetic_data_path testing_output/postprocessed/resample-synthetic-phecodexm.npy \
    --real_eval_data_path testing_output/mimic3-real-phecodexm.npy \
    --output_dir testing_output/evaluation/resample-mimic3-evaluation.json

python -m synthEHRella.run_evaluation pbr \
    --synthetic_data_path testing_output/postprocessed/pbr-synthetic-phecodexm.npy \
    --real_eval_data_path testing_output/mimic3-real-phecodexm.npy \
    --output_dir testing_output/evaluation/pbr-mimic3-evaluation.json