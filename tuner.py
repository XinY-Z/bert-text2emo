from nni.experiment import Experiment
from config import config


# create and setup experiment
experiment = Experiment('local')
experiment.config.experiment_name = config['y']
experiment.config.trial_command = 'python3 train.py'
experiment.config.trial_code_directory = '.'
experiment.config.training_service.use_active_gpu = True
experiment.config.trial_gpu_number = 4
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 100
experiment.config.max_experiment_duration = '72h'
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.search_space = {
    'lr': {'_type': 'loguniform', '_value': [1e-5, 1e-3]},
    'model_name': {'_type': 'choice', '_value': ['facebookAI/roberta-large',
                                                 'microsoft/deberta-large']},
    'optimizer': {'_type': 'choice', '_value': ['Adam', 'AdamW']},
    'bert_output': {'_type': 'choice', '_value': ['pooler', 'pooled_mean', 'last_4']},
    'lora_dropout': {'_type': 'uniform', '_value': [.0, .4]},
    'lora_r': {'_type': 'choice', '_value': [8, 16, 32]},
    'lora_alpha': {'_type': 'choice', '_value': [8, 16, 32]},
    'lora_bias': {'_type': 'choice', '_value': ['none', 'lora_only', 'all']}
}


if __name__ == '__main__':
    experiment.run(port=8080)
    input('Press any key to exit...')
    experiment.stop()
