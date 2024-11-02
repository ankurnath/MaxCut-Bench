

# python evaluation.py --algorithm TS --test_distribution WattsStrogatz_800vertices_unweighted
# python evaluation.py --algorithm TS --test_distribution WattsStrogatz_800vertices_weighted
# python evaluation.py --algorithm TS --test_distribution HomleKim_800vertices_weighted
# python evaluation.py --algorithm TS --test_distribution HomleKim_800vertices_unweighted
# python evaluation.py --algorithm TS --test_distribution BA_800vertices_weighted
# python evaluation.py --algorithm TS --test_distribution BA_800vertices_unweighted

python evaluation.py --algorithm EO --test_distribution WattsStrogatz_800vertices_unweighted
python evaluation.py --algorithm EO --test_distribution WattsStrogatz_800vertices_weighted
python evaluation.py --algorithm EO --test_distribution HomleKim_800vertices_weighted
python evaluation.py --algorithm EO --test_distribution HomleKim_800vertices_unweighted
python evaluation.py --algorithm EO --test_distribution BA_800vertices_weighted
python evaluation.py --algorithm EO --test_distribution BA_800vertices_unweighted