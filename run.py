import argparse
import os
import torch
import numpy as np
import pickle
from pymoo.optimize import minimize
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_algorithm, get_decision_making, get_decomposition
from pymoo.visualization.scatter import Scatter

from config import get_config
from problem import GenerationProblem
from operators import get_operators

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

iteration = 0
conf = AttrDict()

def run(model, target):
    global iteration
    global conf
    iteration = 0
    conf = AttrDict()
    conf.update({
        "device": "cuda",
        "config": model,
        "generations": 525,
        "save_each": 25,
        "tmp_folder": "./tmp",
        "target": target
    })

    conf.update(get_config(conf["config"]))

    def save_callback(algorithm):
        global iteration
        global conf

        iteration += 1
        if iteration % conf.save_each == 0 or iteration == conf.generations:
            if conf.problem_args["n_obj"] == 1:
                sortedpop = sorted(algorithm.pop, key=lambda p: p.F)
                X = np.stack([p.X for p in sortedpop])  
            else:
                X = algorithm.pop.get("X")
            
            ls = conf.latent(conf)
            ls.set_from_population(X)

            with torch.no_grad():
                generated = algorithm.problem.generator.generate(ls, minibatch=conf.batch_size)
                if conf.task == "txt2img":
                    ext = "jpg"
                elif conf.task == "img2txt":
                    ext = "txt"
                name = "genetic-it-%d.%s" % (iteration, ext) if iteration < conf.generations else "genetic-it-final.%s" % (ext, )
                algorithm.problem.generator.save(generated, os.path.join(conf.tmp_folder, name))

    problem = GenerationProblem(conf)
    operators = get_operators(conf)

    if not os.path.exists(conf.tmp_folder): os.mkdir(conf.tmp_folder)

    algorithm = get_algorithm(
        conf.algorithm,
        pop_size=conf.pop_size,
        sampling=operators["sampling"],
        crossover=operators["crossover"],
        mutation=operators["mutation"],
        eliminate_duplicates=True,
        callback=save_callback,
        **(conf.algorithm_args[conf.algorithm] if "algorithm_args" in conf and conf.algorithm in conf.algorithm_args else dict())
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", conf.generations),
        save_history=False,
        verbose=True,
    )


    pickle.dump(dict(
        X = res.X,
        F = res.F,
        G = res.G,
        CV = res.CV,
    ), open(os.path.join(conf.tmp_folder, "genetic_result"), "wb"))

    if conf.problem_args["n_obj"] == 2:
        plot = Scatter(labels=["similarity", "discriminator",])
        plot.add(res.F, color="red")
        plot.save(os.path.join(conf.tmp_folder, "F.jpg"))


    if conf.problem_args["n_obj"] == 1:
        sortedpop = sorted(res.pop, key=lambda p: p.F)
        X = np.stack([p.X for p in sortedpop])
    else:
        X = res.pop.get("X")

    ls = conf.latent(conf)
    ls.set_from_population(X)

    torch.save(ls.state_dict(), os.path.join(conf.tmp_folder, "ls_result"))

    if conf.problem_args["n_obj"] == 1:
        X = np.atleast_2d(res.X)
    else:
        try:
            result = get_decision_making("pseudo-weights", [0, 1]).do(res.F)
        except:
            print("Warning: cant use pseudo-weights")
            result = get_decomposition("asf").do(res.F, [0, 1]).argmin()

        X = res.X[result]
        X = np.atleast_2d(X)

    ls.set_from_population(X)

    with torch.no_grad():
        generated = problem.generator.generate(ls)

    if conf.task == "txt2img":
        ext = "jpg"
    elif conf.task == "img2txt":
        ext = "txt"

    problem.generator.save(generated, os.path.join(conf.tmp_folder, "output.%s" % (ext)))
