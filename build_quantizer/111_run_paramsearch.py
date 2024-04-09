from nni.experiment import Experiment

experiment = Experiment("local")

search_space = {
    "max_beam_size": {"_type": "choice", "_value": [4, 6, 8, 12, 16]},
    "use_beam_LUT": {"_type": "choice", "_value": [0, 1]},
    "num_codebooks": {"_type": "choice", "_value": [1, 2, 3, 4]},
    "codebook_bits": {"_type": "choice", "_value": [7, 8, 9]},
}

experiment.config.trial_command = "python 110_build_additive_quantizer_paramsearch.py"
experiment.config.trial_code_directory = "."
experiment.config.search_space = search_space
experiment.config.tuner.name = "TPE"
experiment.config.tuner.class_args["optimize_mode"] = "minimize"
experiment.config.max_trial_number = 40
experiment.config.trial_concurrency = 1

experiment.run(8080)
