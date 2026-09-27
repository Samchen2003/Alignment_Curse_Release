import os
import faiss
import numpy as np

class Retrieval():
    def __init__(self, text_embedding_model, logger):
        """
        :param text_embedding_model: Any model with an `.encode(text) -> np.array` method.
        :param logger: A logger instance for logging.
        """
        self.model = text_embedding_model
        self.logger = logger

    def embed(self, text):
        """Encode text using the provided model."""
        embedding = self.model.encode(text)
        try:
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            else:
                embedding = embedding.astype(np.float32)
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to convert embedding to np.array: {e}", exc_info=True)
            return None

    def pop(self, library, query, k=5):
        """
        :param library:
            {
              "strategy_name": {
                "Strategy": str,
                "Definition": str,
                "Example": str,
                "Score": float,
                "Embeddings": list[np.Array]  # Each np.Array is a single embedding
              },
              ...
            }
        :param query: The query string to retrieve strategies for.
        :param k: The maximum number of final strategies to return, depending on the logic.
        :return:
            A list of strategy dictionaries (up to `k`),
            or a single-element list if a high-score strategy was found,
            or an empty list if no suitable strategy meets conditions.
        """
        self.logger.info(f"Searching for strategies similar to: {query}")

        query_embedding = self.embed(query)
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        else:
            query_embedding = query_embedding.astype(np.float32)

        all_embeddings = []
        all_scores = []
        all_example = []
        reverse_map = [] 
        for s_name, s_info in library.items():
            emb_list = s_info["Embeddings"]
            score_list = s_info["Score"]
            example_list = s_info["Example"]
            for i in range(len(emb_list)):
                emb = emb_list[i]
                score = score_list[i]
                example = example_list[i]
                if not isinstance(emb, np.ndarray):
                    emb = np.array(emb, dtype=np.float32)
                else:
                    emb = emb.astype(np.float32)

                all_embeddings.append(emb)
                all_scores.append(score)
                all_example.append(example)
                reverse_map.append(s_name)  

        if len(all_embeddings) == 0:
            self.logger.error("No embeddings found in the library.")
            return True, []

        all_embeddings = np.array(all_embeddings, dtype=np.float32)
        dim = all_embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)  
        index.add(all_embeddings)

        num_to_retrieve = len(all_embeddings)
        distances, indices = index.search(query_embedding.reshape(1, dim), num_to_retrieve)

        distances, indices = distances[0], indices[0]

        max_strategies_to_collect = 2 * k
        seen_strategies = set()
        retrieved_strategies = {}

        for dist, idx in zip(distances, indices):
            s_name = reverse_map[idx]
            s_score = all_scores[idx]
            s_example = all_example[idx]
            if s_name not in seen_strategies:
                seen_strategies.add(s_name)
                s_info = library[s_name]
                retrieved_strategies[s_name] = {
                    "Strategy": s_info["Strategy"],
                    "Definition": s_info["Definition"],
                    "Example": s_example,
                    "Score": s_score
                }
            else:
                prev_score = retrieved_strategies[s_name]["Score"]
                retrieved_strategies[s_name]["Score"] = (prev_score+s_score)/2
                if prev_score < s_score:
                    retrieved_strategies[s_name]["Example"] = s_example
            if len(list(retrieved_strategies.keys())) >= max_strategies_to_collect:
                break

        final_retrieved_strategies = []
        final_ineffective_strategies = []
        for final_s_name, final_s_info in retrieved_strategies.items():
            if final_s_info["Score"] >= 5:
                new_dict = {key: value for key, value in final_s_info.items() if key != 'Score'}
                final_retrieved_strategies = [new_dict]
                break
            elif 2 <= final_s_info["Score"] < 5:
                new_dict = {key: value for key, value in final_s_info.items() if key != 'Score'}
                final_retrieved_strategies.append(new_dict)
                if len(final_retrieved_strategies) >= k:
                    break
            else:
                new_dict = {key: value for key, value in final_s_info.items() if key != 'Score'}
                final_ineffective_strategies.append(new_dict)
        if final_retrieved_strategies:
            return True, final_retrieved_strategies
        if len(final_ineffective_strategies) > k:
            return False, final_ineffective_strategies[:k]
        return False, final_ineffective_strategies

