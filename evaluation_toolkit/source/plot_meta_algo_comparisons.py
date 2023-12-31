# -*- coding: utf-8 -*-

############################################################################
#  This file is part of the 4D Light Field Benchmark.                      #
#                                                                          #
#  This work is licensed under the Creative Commons                        #
#  Attribution-NonCommercial-ShareAlike 4.0 International License.         #
#  To view a copy of this license,                                         #
#  visit http://creativecommons.org/licenses/by-nc-sa/4.0/.                #
#                                                                          #
#  Authors: Katrin Honauer & Ole Johannsen                                 #
#  Contact: contact@lightfield-analysis.net                                #
#  Website: www.lightfield-analysis.net                                    #
#                                                                          #
#  The 4D Light Field Benchmark was jointly created by the University of   #
#  Konstanz and the HCI at Heidelberg University. If you use any part of   #
#  the benchmark, please cite our paper "A dataset and evaluation          #
#  methodology for depth estimation on 4D light fields". Thanks!           #
#                                                                          #
#  @inproceedings{honauer2016benchmark,                                    #
#    title={A dataset and evaluation methodology for depth estimation on   #
#           4D light fields},                                              #
#    author={Honauer, Katrin and Johannsen, Ole and Kondermann, Daniel     #
#            and Goldluecke, Bastian},                                     #
#    booktitle={Asian Conference on Computer Vision},                      #
#    year={2016},                                                          #
#    organization={Springer}                                               #
#    }                                                                     #
#                                                                          #
############################################################################


from toolkit.utils.option_parser import OptionParser, SceneOps, AlgorithmOps, MetaAlgorithmOps


def main():
    parser = OptionParser([SceneOps(), AlgorithmOps(), MetaAlgorithmOps()])
    scenes, algorithms, meta_algorithms, compute_meta_algos = parser.parse_args()
    scenes = ['cotton', 'dots', 'backgammon']

    # delay imports to speed up usage response
    from toolkit.algorithms import MetaAlgorithm
    from toolkit.evaluations import meta_algo_comparisons

    # gehuageyan
    #compute_meta_algos = False

    if compute_meta_algos and meta_algorithms:
        MetaAlgorithm.prepare_meta_algorithms(meta_algorithms, algorithms, scenes)

    for meta_algorithm in meta_algorithms:
        meta_algo_comparisons.plot(algorithms, scenes, meta_algorithm)


if __name__ == "__main__":
    main()
