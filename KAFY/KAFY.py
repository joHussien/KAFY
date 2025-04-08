

# %%
from math import log10
import h3
from shapely.geometry import LineString, Point
from typing import List,Dict,Optional
from transformers import pipeline,logging,AutoTokenizer, AutoModelForNextSentencePrediction,AutoModelForMaskedLM
import torch
from types import SimpleNamespace
# import ray
import pickle
import itertools
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from itertools import repeat
import time
from bearing import calculate_bearing
# from bearing import calculate_bearing
import numpy as np
from Partitioning import PartitioningModule
import os
import logging as logging_logging
logging_logging.basicConfig(
    level=logging_logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# For reference later
# Print log messages
"""
logging.debug("This is a debug message.")  # Won't print unless level is set to DEBUG
logging.info("This is an info message.")   # Prints this message
logging.warning("This is a warning message.")  # Prints this message
logging.error("This is an error message.")  # Prints this message
logging.critical("This is a critical message.")  # Prints this message
"""
#%%
class KAFY(object):
    
    
    bert_dir= None
    bert_model = None
    h3_clusters = None
    h3_resolution = None
    
    detokenizer = None
    beam_size = None
    use_constraints = None

    operation = None
    project_path = "/speakingTrajectories/KAFYNewVersionDecember/projectExample"

    def __init__(self, bert_dir:str="", detokenizer:str="", 
            beam_size:int=10, beam_normalization:float=0.7,
            use_constraints:bool=False,project_path:str=""):
        self.bert_dir = bert_dir
        # TODO: These params are operation-specific, beam is even more speciic for the Beam search in imputation
        self.detokenizer = getattr(self, detokenizer,None)
        self.beam_size = beam_size
        self.beam_normalization = beam_normalization
        self.use_constraints = use_constraints
        self.classification_models_initlizated = False
        self.imputation_models_initialized = False
        self.project_path = project_path
        self.models_initialized = False #Two trajectories can end up in same cell so no need to reload the model then
        self.partitioning_module = PartitioningModule(self.project_path)
        logging_logging.info("Pipeline Initialized Successfully")

    def get_model_for_operation(
        self,
        trajectory: List[Point],
        operation: str,
        model_name: Optional[str] = "Bert",
    ) -> Optional[str]:
        """
        Finds the enclosing cell for a trajectory, checks if the operation exists in the metadata,
        and retrieves the path to the fine-tuned model.

        Args:
            trajectory (List[Point]): A trajectory represented as a list of shapely.geometry.Point objects.
            operation (str): The operation to look for in the metadata (e.g., "classification").
            model_name (Optional[str]): The name of the model (e.g., "Bert"). If None, retrieves the first available model.

        Returns:
            Optional[str]: The path to the fine-tuned model if it exists, or None otherwise.
        """
        # Initialize the partitioning module
        

        # Get the enclosing cell for the trajectory

        enclosing_cell = self.partitioning_module._find_enclosing_cell_of_trajectory_list([trajectory])
        print(enclosing_cell)
        if not enclosing_cell:
            raise ValueError("No enclosing cell found for the given trajectory.")

        # Access the metadata of the enclosing cell
        metadata = enclosing_cell.get("metadata", {})

        # Check if the operation exists in the metadata
        if operation not in metadata:
            raise ValueError(f"Operation '{operation}' not found in the cell's metadata.")

        # Get the list of models fine-tuned for the operation
        models = metadata[operation]

        # Determine which model to use
        if model_name:
            if model_name not in models:
                raise ValueError(f"Model '{model_name}' not found for operation '{operation}'.")
            key = model_name
        else:
            # Default to the first model if no model name is provided
            key = models[0] if models else None

        if not key:
            raise ValueError(f"No models available for operation '{operation}'.")

        # Construct the model path
        model_path = os.path.join(enclosing_cell["model_path"], key)

        return model_path    
    

    def init_models(self,mode,operation,trajectory:List[Point],model_name:str="Bert"):
        """
        Initializes the model based on the specified mode and operation, and prepares the necessary 
        configurations for the given trajectory.

        This function handles two main modes: "testing" and "training". In "testing" mode, it loads
        the appropriate model based on the operation and trajectory, and prepares it for inference.
        In "training" mode, the function currently doesn't perform any actions but can be extended
        for future use.

        The function also handles model-specific configurations such as the resolution, pipelines,
        and other operational specifics, which vary depending on the operation (e.g., "classification",
        "imputation").

        Args:
            mode (str): The mode in which the function operates. Options are:
                - "testing": Loads the model for inference.
                - "training": Placeholder for model training functionality (not implemented).
            operation (str): The specific operation to be performed. Options include:
                - "classification": Initializes a classification pipeline.
                - "imputation": Initializes a fill-mask pipeline.
            trajectory (List[Point]): A list of GPS trajectory points used to define the model's spatial context.
            model_name (str, optional): The name of the model to be used. Defaults to "Bert".

        Returns:
            None: This function doesn't return anything, but it sets up the model and configurations 
                for subsequent operations.

        Raises:
            ValueError: If the mode or operation is not recognized.

        Example:
            # Initialize the model for classification in testing mode
            self.init_models(mode="testing", operation="classification", trajectory=my_trajectory)

        Notes:
            - In testing mode, the function loads a pre-existing model based on the operation and trajectory.
            - In training mode, the function doesn't perform any actions but serves as a placeholder.
            - The function also supports future extension by adding more pipelines for other operations.
        """
        # Call the paritioning module given the operation and trajectory and return model path at the end
        if mode == "testing":
           self.bert_dir_old = self.bert_dir
           self.bert_dir = self.get_model_for_operation(operation=operation,trajectory=trajectory,model_name=model_name)
           print(self.bert_dir)
           with open(f'{self.bert_dir}/resolution.txt', 'r') as file: 
                    self.h3_resolution = int(file.readlines()[0])
           self.models_initialized = False if self.bert_dir_old!= self.bert_dir else True #if loading new model then models_not_initialized
        elif mode=="training":
            print("I do not what to do here")

        # Inference part itself is here. 
        # If a user wants to add a new operation he has to define the inference sequence here  
        if operation =='classification':
            if not self.models_initialized:
                logging.set_verbosity_error()
                self.bert_model = pipeline('text-classification', model=self.bert_dir)
        elif operation=="imputation":
            if not self.models_initialized:
                logging.set_verbosity_error()
                print(self.bert_dir)
                # model = AutoModelForMaskedLM.from_pretrained(self.bert_dir,from_tf=True)
                # tokenizer = AutoTokenizer.from_pretrained(self.bert_dir,from_tf=True)
                self.bert_model = pipeline('fill-mask', model=self.bert_dir, top_k=self.beam_size, tokenizer =tokenizer)
                if self.detokenizer == self.token2point_cluster_centroid:
                    with open(f'{self.bert_dir}/clusters.pkl', 'rb') as file:
                        self.h3_kmeans = pickle.load(file)

                with open(f'{self.bert_dir}/resolution.txt', 'r') as file: 
                    self.h3_resolution = int(file.readlines()[0])
            
            #In case of Next Sentence Prediction
        elif operation =="prediction":
            if not self.models_initialized:
                logging.set_verbosity_error()
                print(self.bert_dir)
                self.bert_model = AutoModelForNextSentencePrediction.from_pretrained(self.bert_dir)
                print(self.bert_model)
                self.model_tokenizer = AutoTokenizer.from_pretrained(self.bert_dir)
                print("This is the tokenizer")
                print(self.model_tokenizer)
            '''
            import torch

            tokenizer = AutoTokenizer.from_pretrained(self.bert_dir)
            model = BertForNextSentencePrediction.from_pretrained(self.bert_dir)
            #These lines should be in the inference function
            prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            next_sentence = "The sky is blue due to the shorter wavelength of blue light."
           
            '''
            
    
    
    def log(self,message):
        pass
# PARTITIONING is in a seperate file.
# TOKENIZATION
    # Now the operation/query data is giving me the resoltuion, not one defined operation

    def points2tokens(self, points: List[Point], resolution: int) -> List[str]:
        """
        Converts a list of shapely.geometry.Point objects into H3 tokens at a specified resolution.

        Each point represents a geographical location, and the function returns the H3 indices for 
        these points at the given resolution.

        Args:
            points (List[Point]): A list of shapely.geometry.Point objects, where each point 
                                contains latitude (y) and longitude (x) coordinates.
            resolution (int): The H3 resolution level to use for tokenization.

        Returns:
            List[str]: A list of H3 tokens (hexagonal cell indices) corresponding to the input points.
        """
        return [h3.latlng_to_cell(point.y, point.x, resolution) for point in points]

# DETOKENIZATION    
    def token2point_h3_centroid(self, token, previous_point=None):
        # returns centroid of the hexagon
        y, x = h3.cell_to_latlng(token)
        return Point(x,y)

    def token2point_data_centroid(self, token, previous_point=None):
        # return centroid of "all data" in the hexagon
        if token in self.h3_clusters:
            cluster = self.h3_clusters[token]
            x, y = cluster['x'], cluster['y']
            return Point(x,y)
        else:
            return self.token2point_h3_centroid(token, None)

    
    def token2point_cluster_centroid(self, token, previous_point):
        # return centroid of the closest cluster 
        c = self.token2point_data_centroid(token, None)
        
        if token not in self.h3_kmeans:
            return c
        
        if token in self.h3_clusters and self.h3_clusters[token]['current_count'] <= 20:
            return c

        angle = calculate_bearing((previous_point.y, previous_point.x), (c.y, c.x))
        
        m, means = self.h3_kmeans[token]
        x, y, _ = means[m.predict(np.array([angle]).reshape(-1,1))][0]
        return Point(x,y)

# IMPUTATION

    def call_bert(self, part1, part2):

        input = part1 +  ['[MASK]'] + part2
        input = ' '.join(input)
        
        # In this case, I didn't need to do HuggingFace tokenization as this is a pipeline instance

        json = self.bert_model(input)
        
        # TODO: check if API successed and json is not empty
        results = {
            'ok': True,
            'json': json
        }
        return SimpleNamespace(**results)

    def get_constrained_candidates_between_two_points(self, p_from, p_to, factor=1):
        dist = h3.grid_distance(p_from,p_to)
        dist = round(dist * factor)
        ring1 = h3.grid_ring(p_from, dist)
        ring2 = h3.grid_ring(p_to, dist)
        constrained_candidates = set(ring1).intersection(ring2)
        return set(constrained_candidates)

    def beam_search(self, part1, part2, 
                other_segments_imputations = [],
                max_length = None,
                beam_size = None,
                beam_normalization = None):


        
        beam_size = beam_size or self.beam_size
        beam_normalization = beam_normalization or self.beam_normalization   

        self.log(f'beam search is called for the follwoing points: {[part1[-1], part2[0], other_segments_imputations]}')

        # initial empty beam results to build on
        beam_results = [{
            "sequence": [],
            "score": 0,
            "normalized_score": 0,
            "has_no_more_candidates": False
        }]

        org_constined_cand = set(
            self.get_constrained_candidates_between_two_points(
                                        part1[-1], part2[0]))



        for i in range(max_length):
            new_solutions = []
            for result in beam_results:
                if result["has_no_more_candidates"]:
                    # keep the results in our solution list, but no need to try to check for 
                    # new following points since we didn't find any thing in the last step i-1
                    # so we will get the same results if we call again.
                    new_solutions.append(result)
                    continue

                sequence = result["sequence"]
                found_at_least_1_candidate = False


                for gap_placement in range(len(sequence)+1):
                    modified_part1 =  part1 + sequence[:gap_placement]
                    modified_part2 = sequence[gap_placement:] + part2

                    p_from = modified_part1[-1] 
                    p_to = modified_part2[0]

                    if h3.grid_distance(p_from, p_to) <=1:
                        continue

                    # prepare the constrained candidates
                    constrained_cand = org_constined_cand.intersection(
                        self.get_constrained_candidates_between_two_points(p_from, p_to))
                
                    self.log(f"""calling bert as folows: 
                    part 1 {modified_part1}
                    part 2 {modified_part2}
                    """)
                    res = self.call_bert(modified_part1, modified_part2)

                    predictions = res.json
                    

                    # bert will return the top x (x=beam_size because it is intialized to do so in init_models)
                    for p in predictions:
                        candidate = p['token_str']
                        score =  p['score']
                        self.log(f'new candidate {candidate} with score {score}')

                        if candidate == p_from or candidate == p_to:
                            self.log('skipped because it matches the gap boundaries')
                            continue

                        if candidate in sequence:
                            self.log('skipped because it is already in the current sequence')
                            continue
                        

                        if self.use_constraints and candidate not in constrained_cand:
                            self.log('skipped because it is not allowed/ not among the constrained candidates')
                            continue

                        
                        found_at_least_1_candidate = True
                        

                        new_sequence = [*sequence]
                        new_sequence.insert(gap_placement, candidate)

                        new_score = result["score"] + (-1 * log10(score))
                        
                        # normalize by the length
                        new_normalized_score = new_score / (len(new_sequence)**beam_normalization)

                        new_solutions.append({
                            "sequence": new_sequence,
                            "score": new_score,
                            "normalized_score": new_normalized_score,
                            "has_no_more_candidates": False
                        })
                        self.log (f'new solution appended as follow. \n {new_solutions[-1]}')


                # if no more candidate at this step i, then there will be no candidates as weel at i+1
                # so no need to check again at the next time step. we flag has_no_more_candidate as True
                # so the next iteration can see that and skit it. But we need to keep the result so 
                # it doesn't get lost
                if not found_at_least_1_candidate:
                    # No more candidates. Append the current results with updated has_no_more_candidates
                    result["has_no_more_candidates"] = True
                    new_solutions.append(result)
                    self.log (f'No more Candidates for the sequence {sequence}')

            
            self.log(f"beam results at {i} was: {beam_results}")
            
            # beam_results now may include duplicates. we will sort then iterate and keep only distinct solutions 
            
            beam_results = sorted(new_solutions, key=lambda s: s['normalized_score'])
            distinct_results = []
            seen_sequences = []
            for solution in beam_results:
                sequence = set(solution['sequence'])
                if sequence in seen_sequences:
                    continue
                else:
                    distinct_results.append(solution)
                    seen_sequences.append(sequence)
            

            beam_results = distinct_results[:beam_size]
            self.log(f'current beam results at end of step i {i} is : \n {beam_results}')
        
        
        top_result = beam_results[0]
        top_sequence = top_result['sequence']
        top_score = top_result['normalized_score']

        self.log(f'beam finished with score: {top_score}' )
        return top_sequence, top_score

 
    def impute_a_gap(self, input_points, gap_at):
        self.init_models(mode="testing",operation='imputation',trajectory=input_points,model_name="Bert")
        start = time.time()

        imputed_seg_pt = input_points[gap_at : gap_at + 2]
        inferred_seg_pt = []

        h3_input = self.points2tokens(input_points,self.h3_resolution)    
        part1 = h3_input[0:gap_at + 1]
        part2 = h3_input[gap_at + 1:]

        p_from = part1[-1] 
        p_to = part2[0]

        max_length =  2 * h3.grid_distance(p_from, p_to)

        self.log(f"""imputing at gap {gap_at}
        part 1 : {part1}
        part 2 : {part2}
        
        """)

        
        most_likely_sequence, score = self.beam_search(part1, part2, other_segments_imputations=[], max_length=max_length)
        # most_likely_sequence, score = self.beam_search(part1, part2, other_segments_imputations=[])
        
        for h3_token in most_likely_sequence:
            p = self.detokenizer(h3_token, imputed_seg_pt[-2])
            
            # insert the point at imputed_seg_pt[-1], i.e. before the last one
            # because we are imputing between two points. 

            imputed_seg_pt.insert(-1, p)
            inferred_seg_pt.append(p)

        end = time.time()
        imputed_seg_time = (end - start)

        self.log(f'done at gap {gap_at} with score {score}')
        return {
            'imputed_seg_pt': imputed_seg_pt,
            'inferred_seg_pt': inferred_seg_pt,
            'imputed_seg_score': score,
            'imputed_seg_time': imputed_seg_time
        }
       
# CLASSIFICATION 
    def classify_trajectory(self, input_points: List[Point]):
        """
        Classifies a trajectory represented by a list of shapely.geometry.Point objects 
        using a pre-trained BERT classification model.

        The function initializes the classification model, retrieves the appropriate 
        resolution for tokenization, and performs classification on the tokenized trajectory.

        Args:
            input_points (List[Point]): A list of shapely.geometry.Point objects representing 
                                        the trajectory to classify. Each point contains 
                                        latitude (y) and longitude (x) coordinates.

        Returns:
            dict: The classification results as output by the BERT model pipeline.
        """
        # This will initalize the pipeline with the appropiate model to be used to classify this trajectory using the Partitioning module
        self.init_models(mode="testing",operation='classification',trajectory=input_points,model_name="Bert")        
        # Tokenize the trajectory
        tokenized_trajectory = ' '.join(self.points2tokens(input_points,self.h3_resolution))
        # In this case, I didnot need to do HuggingFace tokenization as this is a pipeline instance
        classification_results = self.bert_model(tokenized_trajectory)
        return classification_results 
# NEXT TRAJECTORY PREDICTION     
    def predict_is_next_trajectory(self, trajectory_1: List[Point], trajectory_2: List[Point]):
        """
        Predicts whether the second trajectory is the next in sequence based on a comparison between 
        two given trajectories using a fine-tuned BERT model.

        This function performs the prediction by tokenizing the two input trajectories and passing 
        them through a preloaded BERT model in "prediction" mode. It uses HuggingFace's tokenization 
        and model inference process to compute the logits for a binary classification (0 or 1), 
        indicating if the second trajectory is the next in the sequence.

        Args:
            trajectory_1 (List[Point]): A list of GPS points representing the first trajectory.
            trajectory_2 (List[Point]): A list of GPS points representing the second trajectory.

        Returns:
            int: Returns 1 if the second trajectory is predicted to be the next, and 0 otherwise.
            
        Raises:
            AssertionError: If the prediction output does not match the expected assertion.

        Example:
            # Predict if the second trajectory is the next after the first trajectory
            result = self.predict_is_next_trajectory(trajectory_1, trajectory_2)

        """
        # Initialize the model for prediction
        self.init_models(mode="testing", operation='prediction', trajectory=trajectory_1, model_name="Bert")

        # Tokenize both trajectories using the H3 resolution
        tokenized_trajectory_1 = ' '.join(self.points2tokens(trajectory_1, self.h3_resolution))
        tokenized_trajectory_2 = ' '.join(self.points2tokens(trajectory_2, self.h3_resolution))

        # Use the model's tokenizer to encode the two tokenized trajectories
        encoding = self.model_tokenizer(tokenized_trajectory_1, tokenized_trajectory_2, return_tensors="pt")

        # Pass the encoding through the model to get the logits
        outputs = self.bert_model(**encoding, labels=torch.LongTensor([1]))
        logits = outputs.logits

        # Return 1 if the second trajectory is predicted as the next, otherwise return 0
        return 1 if logits[0, 1] > logits[0, 0] else 0
    # %%