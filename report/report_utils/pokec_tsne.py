
from turtle import back
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import re
from itertools import product
from pathlib import Path

from evaluation import LabelPropagationEvaluation

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


class PokecTSNE():
    def process_tsne(self, 
                     experiment_name: str, 
                     graph_number: int,
                     emb_graph_dir: Path,
                     tsne_dict: dict, 
                     attributes: list,
                     baseline: bool,
                     biased: bool,
                     df_filtered_attributes: pd.DataFrame,
                     tsne_plots_dir: Path,
                     ):

                    # Creating all possible combinations of p and q
                    pq_combinations = [f"p_{p}_q_{q}" 
                                       for p, q 
                                        in product(
                                            tsne_dict['p'], 
                                            tsne_dict['q']
                                            )
                                        ]
                    # Creating all possible combinations of p, q, alpha, and exp
                    pqxe_combinations = [f"_alpha_{alpha}_exponent_{exp}_p_{p}_q_{q}" 
                                        for p, q, alpha, exp 
                                            in product(
                                                tsne_dict['p'], 
                                                tsne_dict['q'], 
                                                tsne_dict['alpha'], 
                                                tsne_dict['exponent'],
                                                )
                                        ]
                    print("PQ COMBINATIONS", pq_combinations)
                    print("PQXE COMBINATIONS", pqxe_combinations)

                    attributes = ["AGE"]

                    for emb_file in emb_graph_dir.iterdir():
                        # print("EMB FILE", emb_file)
                        if emb_file.name.startswith("p_") and baseline:
                            for comb in pq_combinations:
                                # Modified matching to also check for "_" after the combination
                                print("EMB FILE NAME", emb_file.name)
                                if re.search(f'{comb}', emb_file.name):
                                    for attr in attributes:
                                        print(f'Found {comb} in {emb_file.name} for experiment {experiment_name} graph {graph_number}')
                                        self.compute_tsne(emb_file_path=emb_file,
                                                        plotting_path=tsne_plots_dir / f"baseline_{experiment_name}_{graph_number}_{comb}_other_{attr}.pdf",
                                                        df_filtered_attributes=df_filtered_attributes,
                                                        other_attribute=attr,
                                                        )
                        if emb_file.name.startswith("pre") and biased:
                            for comb in pqxe_combinations:
                                if re.search(f'{comb}([_])', emb_file.name):
                                    for attr in attributes:
                                        if emb_file.name.endswith(f"{attr}.emb.gz"):
                                            print(f'Found {comb} in {emb_file.name} for experiment {experiment_name} graph {graph_number}')
                                            self.compute_tsne(emb_file_path=emb_file,
                                                            plotting_path=tsne_plots_dir / f"{experiment_name}_{graph_number}_{comb}_other_{attr}.pdf",
                                                            df_filtered_attributes=df_filtered_attributes,
                                                            other_attribute=attr,
                                                            )



    def compute_tsne(self, 
                     emb_file_path: Path, 
                     plotting_path: Path, 
                     df_filtered_attributes: pd.DataFrame, 
                     other_attribute: str, 
                     ) -> None:

        #print("EMB FILE", emb_file_path)
        # print("PLOTTING PATH", plotting_path)
        # print("DF FILTERED ATTRIBUTES", df_filtered_attributes.head())


        # Create dictionaries with user_id as key and the attribute as value
        d_sens_attr_0 = df_filtered_attributes.set_index('user_id')['label_AGE'].to_dict()
        d_sens_attr_1 = df_filtered_attributes.set_index('user_id')['label_region'].to_dict()
                
        
        d_emb, dim = LabelPropagationEvaluation.read_embeddings(embedding_path=emb_file_path)

        # print("D EMB", len(d_emb), "DIM", dim)
        # print("SENS ATTR 0", "SENS ATTR 1", len(d_sens_attr_1))
        # print("Other Attribute", other_attribute, "DF sensattr 0", 
        #         {k: d_sens_attr_0[k] for k in list(d_sens_attr_0)[:10]})

        assert len(d_emb) == len(d_sens_attr_0)
        assert len(d_emb) == len(d_sens_attr_1)

        d_sens_attr = d_sens_attr_0 if other_attribute == "region" \
                                        else d_sens_attr_1
        sens_count = np.unique(list(d_sens_attr_0.values())).shape[0] if other_attribute == "region" \
                                                                        else np.unique(list(d_sens_attr_1.values())).shape[0]
        # print("SENS COUNT", sens_count)
        n = len(d_emb)

        print("Computing TSNE for", plotting_path)
       

        X = np.zeros([n, dim])
        z = np.zeros([n])
        for i, id in enumerate(d_emb):
            X[i, :] = np.array(d_emb[id])
            z[i] = d_sens_attr[id]

        X_emb = TSNE(n_components=2, 
                     learning_rate='auto', 
                     n_jobs=-1,
                     init='pca',
                     verbose=1,
                     ).fit_transform(X)  # init='pca'

        
        # colors = ['b','tab:orange', 'g', 'xkcd:lavender', 'c', 'm', 'r', 'k', 'w',  'tab:purple', 'tab:pink', 'tab:gray', 
        #             'tab:olive', 'tab:cyan', 'xkcd:lightgreen', 'xkcd:salmon',  'xkcd:teal', 
        #             'xkcd:mustard', 'xkcd:brick']

        # https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
        #12 groups 
        # colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

        #https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=4
        # 4 groups 
        colors = ['#d7191c','#abd9e9','#2c7bb6', '#fdae61']

        # colors = [
        #     "#e6194b",  # Red
        #     "#3cb44b",  # Greensemi
        #     "#ffe119",  # Yellow
        #     "#0082c8",  # Blue
        #     "#f58231",  # Orange
        #     "#911eb4",  # Purple
        #     "#46f0f0",  # Cyan
        #     "#f032e6",  # Magenta
        #     "#d2f53c",  # Lime
        #     "#fabebe",  # Pink
        #     "#008080",  # Teal
        #     "#e6beff",  # Lavender
        #     "#aa6e28",  # Brown
        #     "#fffac8",  # Beige
        # ]


        # print(f'Count of different sens_attributes is {sens_count}.')
        for i in range(sens_count):
            X_emb_color = X_emb[z == i, :]
            plt.scatter(X_emb_color[:, 0], X_emb_color[:, 1], color=colors[i], s=3, alpha=0.5)

        # Turn off the axis
        plt.axis('off')

        # Change figure border color and face color to remove the black border
        plt.gcf().set_edgecolor('white') 
        plt.gcf().set_facecolor('white')  # Set to your background color


        print(f"Saving tsne plot to {plotting_path}")

        plt.savefig(plotting_path, bbox_inches='tight')

        plt.clf()
