
from math import e
import re
from unittest import result
import pandas as pd
from collections import defaultdict

from pathlib import Path

from data_utils.graph.pokec_graph import PokecGraph
from report_utils import PokecConfusion, PokecCombinedReports, PokecTSNE


def plot_tsne(d_experiments: dict, 
              tsne_dict: dict,
              plotting_dir: Path,
              experiments_list: list,
              graph_nrs: list,
              attributes: list,
              baseline: bool,
              biased: bool,
              ):
    tsne_plots_dir = plotting_dir / "tsne_plots"
    tsne_plots_dir.mkdir(parents=True, exist_ok=True)
    for ex in experiments_list:
        resources_dir = d_experiments[ex]["Experiment_dir"] / "resources" 
        if resources_dir.exists():
            for emb_graph_dir in resources_dir.iterdir():
                if emb_graph_dir.is_dir():
                    graph_number = int(emb_graph_dir.name.split('_')[-1])
                    if graph_number in graph_nrs:
                        filtered_attributes_file = resources_dir / emb_graph_dir / "filtered_attributes.csv"
                        if filtered_attributes_file.exists():
                            df_filtered_attributes = pd.read_csv(filtered_attributes_file, usecols=["user_id", "label_region", "label_AGE"])
                        else:
                            raise Exception(f"Could not find filtered attributes for graph {graph_number}")
                        emb_graph_dir = emb_graph_dir / "embeddings"
                        tsne = PokecTSNE()
                        print(f"Computing tsne-plots for experiment {ex} graph {graph_number}")
                        tsne.process_tsne(experiment_name=ex,
                                          graph_number=graph_number,
                                          emb_graph_dir=emb_graph_dir,
                                          tsne_dict=tsne_dict,
                                          attributes=attributes,
                                          baseline=baseline,
                                          biased=biased,
                                          df_filtered_attributes=df_filtered_attributes,
                                          tsne_plots_dir=tsne_plots_dir,
                                          )
                    
    


def create_confusion_reports(d_experiments: dict, 
                                 d_graph_filtered_attributes: dict, 
                                 report_dir: Path,
                                 experiments_list: list, 
                                 graph_nrs: list, 
                                 ):
    # if (report_dir / "confusion_reports").exists(): # TODO implement a check for the completed contents of the folder
    #     # delete the folder reports/confusion_reports if exists to avoid partial completion
    #     for file in (report_dir / "confusion_reports").iterdir():
    #         file.unlink()
    #     (report_dir / "confusion_reports").rmdir()
    for ex in experiments_list:
        for g_nr in graph_nrs:
            if g_nr in d_experiments[ex]["Graph_nrs"]:
                graph_results_dir = d_experiments[ex]["Experiment_dir"] / "results" / f"graph_{g_nr}"
                # print(f"Graph results dir: {graph_results_dir}")
                # print(f"Graph results dir content: {list(graph_results_dir.iterdir())}")
                if graph_results_dir.exists():
                    pokec_confusion = PokecConfusion()
                    exp_graph_name = str(f"{ex} {g_nr}")
                    graph_name = graph_results_dir.name
                    df_filtered_attributes = d_graph_filtered_attributes.get(ex).get(f"Graph_nr_{g_nr}")
                    print(f"Graph dir: {graph_results_dir}, \n \
                            exp_graph_name: {exp_graph_name}, \n \
                            graph_name: {graph_name}")
                    pokec_confusion.process_results(graph_results_dir=graph_results_dir, 
                                                    report_dir=report_dir, 
                                                    df_filtered_attributes=df_filtered_attributes, 
                                                    graph_name=graph_name, 
                                                    exp_graph_name=exp_graph_name, 
                                                    )


def create_combined_reports(d_experiments: dict,
                            report_dir: Path, 
                            graph_nrs: list, 
                            ):   
    if (report_dir / "confusion_reports").exists(): 
        for ex in d_experiments.keys():
            pokec_combined = PokecCombinedReports(report_dir=report_dir)  
            pokec_combined.run_averaging_df(experiment_name=ex, 
                                            graph_list=graph_nrs,
                                            )


def build_analyze_graph(d_experiments: dict,
                            experiments_list: list, 
                            graph_nrs: list,
                            report_dir: Path,
                            attributes: list,
                            ):
    for ex in experiments_list:
            for g_nr in graph_nrs:
                if g_nr in d_experiments[ex]["Graph_nrs"]:
                    graph_dir = d_experiments[ex]["Experiment_dir"] / "resources" / f"graph_dir_{g_nr}"
                    if graph_dir.exists():
                        exp_graph_name = (f"{ex} {g_nr}")
                        for file in graph_dir.iterdir():
                            if file.name == "filtered_attributes.csv":
                                attributes_file = file
                                df_attr = pd.read_csv(attributes_file)
                                df_attr = df_attr.set_index('user_id')
                            elif file.name == "filtered_edgelist.txt":
                                edgelist_file = file
                                df_el = pd.read_csv(edgelist_file, sep=" ", header=None)
                        PokecGraph.graph_visual_analyze(df_el = df_el, 
                                                        df_attr = df_attr, 
                                                        report_dir=report_dir, 
                                                        attributes=attributes,
                                                        exp_graph_name=exp_graph_name,\
                                                        )
                        print(f"Graph {g_nr} of experiment {ex} analyzed \
                                    \n and saved to {report_dir} / graph_plots and / graph_specs")


def get_filtered_attributes(d_experiments: dict,
                            experiments_list: list, 
                            graph_nrs: list,
                            attributes: list,
                            ):
    d_graph_filtered_attributes = defaultdict(dict)
    for ex in experiments_list:
        for g_nr in graph_nrs:
            # check if the dictionary entry for graph nr exists
            if g_nr in d_experiments[ex]["Graph_nrs"]:
                resources_dir = d_experiments[ex]["Experiment_dir"] / "resources" / f"graph_dir_{g_nr}"
                if resources_dir.exists():
                    attributes_file = resources_dir / "filtered_attributes.csv"
                    if attributes_file.exists():
                        df_attr = pd.read_csv(attributes_file, usecols=["user_id", *attributes])
                        d_graph_filtered_attributes[ex][f"Graph_nr_{g_nr}"] = df_attr
                    else:
                        print(f"Attributes file does not exist: {attributes_file}")
                else:
                    print("Graph dir does not exist: {graph_dir}")
            else:
                print(f"Graph {g_nr} of experiment {ex} not found in the dictionary")

    return d_graph_filtered_attributes


def create_reporting_process_dict(data_dir: Path, 
                                  sub_ex_name_start: str, 
                                  ):
    d_experiments = defaultdict(dict)

    experiment_dirs = [ex for ex in data_dir.iterdir() \
                if ex.is_dir() and ex.name.startswith(sub_ex_name_start)]

    for i, ex_dir in enumerate(experiment_dirs):
        resources_dir = ex_dir / "resources"
        experiment_name = ex_dir.name.split("_")[0] + " " + ex_dir.name.split("_")[1]
        # d_experiments[experiment_name] = dict()
        if i == 0:
            print("Experiment/Graph name(s) found: ")
        print(experiment_name)
        resources = [_ for _ in resources_dir.iterdir() if _.is_dir()]
        graph_nrs = list()
        print(resources_dir)
        for j, g in enumerate(resources):
            graph_nr = int(g.name.split("_")[2])
            graph_nrs.append(graph_nr)
            # print(graph_nr)
            if j==0:  
                print("With")
            print(f"  Graph {graph_nr}")    
        d_experiments[experiment_name]["Graph_nrs"] = graph_nrs
        d_experiments[experiment_name]["Experiment_dir"] = ex_dir

    return d_experiments


