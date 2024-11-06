python -m synthEHRella.run_postprocessing corgan \
    --data_path testing_output/corgan/synthetic-1000.npy \
    --output_path testing_output/postprocessed/corgan-synthetic-phecodexm.npy

python -m synthEHRella.run_postprocessing ehrdiff \
    --data_path testing_output/ehrdiff/samples/all_x.npy \
    --output_path testing_output/postprocessed/ehrdiff-synthetic-phecodexm.npy

python -m synthEHRella.run_postprocessing medgan \
    --data_path testing_output/medgan/synthetic.npy \
    --output_path testing_output/postprocessed/medgan-synthetic-phecodexm.npy

python -m synthEHRella.run_postprocessing plasmode \
    --data_path testing_output/plasmode \
    --output_path testing_output/postprocessed/plasmode-synthetic-phecodexm.npy

python -m synthEHRella.run_postprocessing promptehr \
    --data_path testing_output/promptehr.npy \
    --output_path testing_output/postprocessed/promptehr-synthetic-phecodexm.npy

python -m synthEHRella.run_postprocessing synthea \
    --data_path testing_output/synthea/csv/conditions.csv \
    --output_path testing_output/postprocessed/synthea-synthetic-phecodexm.npy

python -m synthEHRella.run_postprocessing vae \
    --data_path testing_output/vae/synthetic.npy \
    --output_path testing_output/postprocessed/vae-synthetic-phecodexm.npy    

python -m synthEHRella.run_postprocessing resample \
    --data_path testing_output/resample/resample-synthetic.npy \
    --output_path testing_output/postprocessed/resample-synthetic-phecodexm.npy

python -m synthEHRella.run_postprocessing pbr \
    --data_path testing_output/pbr/pbr-synthetic.npy \
    --output_path testing_output/postprocessed/pbr-synthetic-phecodexm.npy