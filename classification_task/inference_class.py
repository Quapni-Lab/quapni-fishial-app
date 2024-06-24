import torch.nn as nn
import numpy as np
import logging
import torch
import json

from PIL import Image
from torchvision import transforms


def read_json(path_to_json):
    with open(path_to_json) as f:
        return json.load(f)
    
def get_results(output):
        top_1, top_1_val = None, 10e9
        top_median, top_median_val = None, 10e9

        for i in output:
            if top_1_val > output[i]['top_1']:
                top_1 = i
                top_1_val = output[i]['top_1']

            if top_median_val > output[i]['median']:
                top_median = i
                top_median_val = output[i]['median']
        return [top_1, top_median]
    
class EmbeddingClassifier:
    def __init__(self, model_path, data_set_path, data_id_path, device='cpu', THRESHOLD = 8.84):
        self.device = device
        self.THRESHOLD = THRESHOLD
        self.indexes_of_elements = read_json(data_id_path)
        self.softmax = nn.Softmax(dim=1)
        
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.to(device)
        
        self.loader = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.data_base = torch.load(data_set_path).to(device)
        logging.info("[INIT][CLASSIFICATION] Initialization of classifier was finished")
                
    def __inference(self, image, top_k = 15):
        logging.info("[PROCESSING][CLASSIFICATION] Getting embedding for a single detection mask")
        
        dump_embed, fc_output = self.model(image.unsqueeze(0).to(self.device))
        
        logging.info("[PROCESSING][CLASSIFICATION] Classification by Full Connected layer for a single detection mask")
        classes, scores = self.__classify_fc(fc_output)
        
        logging.info("[PROCESSING][CLASSIFICATION] Classification by embedding for a single detection mask")
        output_by_embeddings = self.__classify_embedding(dump_embed[0], top_k)
        
        logging.info("[PROCESSING][CLASSIFICATION] Beautify output for a single detection mask")
        result = self.__beautifier_output(output_by_embeddings, self.indexes_of_elements['categories'][str(classes[0].item())]['name'])
        result = [[self.__get_species_name(record[0]), record] for record in result]
        return result
    
    def __beautifier_output(self, output_by_embeddings, classification_label):
            
        dict_results = {}
        for i in output_by_embeddings:
            if i[0]['name'] in dict_results:
                dict_results[i[0]['name']]['values'].append(i[2].item())
                dict_results[i[0]['name']]['annotations'].append(i[1])
            else:
                dict_results.update({i[0]['name']: {
                    'values': [i[2].item()],
                    'annotations': [i[1]]
                }})

        for i in dict_results:
            dict_results[i].update({'top_1': dict_results[i]['values'][0]})
            dict_results[i].update({'annotation': dict_results[i]['annotations'][0]})
            dict_results[i].update({'median': np.median(dict_results[i]['values'])})
            del dict_results[i]['values']
            del dict_results[i]['annotations']

        labels = get_results(dict_results)
        labels = list(set(labels))

        for result in list(dict_results.keys()):
            if result not in labels:
                del dict_results[result]
            else:
                mean_distance = (dict_results[result]['top_1'] + dict_results[result]['median'])/2
                dict_results[result]['dist'] = mean_distance
                dict_results[result]['conf'] = round(self.__get_confidence(mean_distance), 3)
                logging.info(f"[PROCESSING][CLASSIFICATION] the threshold |{mean_distance}| has been recalculated to |{dict_results[result]['conf']}|")
        results = [[label, dict_results[label]['conf'], dict_results[label]['annotation']] for label in dict_results]
        logging.info("[PROCESSING][CLASSIFICATION] Classification by embedding was finished successfuly")

        if classification_label not in labels:
            logging.info("[PROCESSING][CLASSIFICATION] Append into output classification result by FC - layer")
            results.append([classification_label, 0.1, [None, None, None]])
        else:
            logging.info("[PROCESSING][CLASSIFICATION] Output from FC layer exist in Embedding results")
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results
    
    def __get_confidence(self, dist):
        min_dist = 4.2
        max_dist = self.THRESHOLD
        delta = max_dist - min_dist
        return 1.0 - (max(min(max_dist, dist), min_dist) - min_dist) / delta
    
    def inference_numpy(self, img, top_k=10):
        image = Image.fromarray(img)
        image = self.loader(image)
        
        return self.__inference(image, top_k)
    
    def batch_inference(self, imgs):
        batch_input = []
        for idx in range(len(imgs)):
            image = Image.fromarray(imgs[idx])
            image = self.loader(image)
            batch_input.append(image)

        batch_input = torch.stack(batch_input)
        dump_embeds, class_ids = self.model(batch_input)
        
        logging.info("[PROCESSING][CLASSIFICATION] Classification by Full Connected layer for a single detection mask")
        classes, scores = self.__classify_fc(class_ids)
       
        outputs = []
        for output_id in range(len(classes)):

            logging.info("[PROCESSING][CLASSIFICATION] Classification by embedding for a single detection mask")
            output_by_embeddings = self.__classify_embedding(dump_embeds[output_id])
            result = self.__beautifier_output(output_by_embeddings, self.indexes_of_elements['categories'][str(classes[output_id].item())]['name'])
            result = [[self.__get_species_name(record[0]), record] for record in result]
            
            outputs.append(result)
            
        return outputs
    
    def __classify_fc(self, output):
        acc_values = self.softmax(output)
        class_id = torch.argmax(acc_values, dim=1)
        #print(f"Recognized species id {class_id} with liklyhood: {acc_values[0][class_id]}")
        return class_id, acc_values

    def __classify_embedding(self, embedding, top_k = 15):
        diff = (self.data_base - embedding).pow(2).sum(dim=1).sqrt()
        val, indi = torch.sort(diff)
        class_lib = [[self.indexes_of_elements['list_of_ids'][indiece], diff[indiece]] for indiece in indi[:top_k]]
        class_lib = [[self.indexes_of_elements['categories'][str(rec[0][0])],rec[0], rec[1]] for rec in class_lib]
        return class_lib

    
    def __get_species_name(self, category_name):
        for i in self.indexes_of_elements['categories']:
            if self.indexes_of_elements['categories'][i]['name'] == category_name:
                return self.indexes_of_elements['categories'][i]['species_id']
