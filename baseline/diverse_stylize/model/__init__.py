import importlib


def find_model_using_name(model_name):
    model_filename = "model." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('No model named %s that matches in lowercase in %s.py' % (target_model_name, model_filename))
        exit(0)

    return model


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print('Model [%s] created' % type(instance).__name__)
    return instance