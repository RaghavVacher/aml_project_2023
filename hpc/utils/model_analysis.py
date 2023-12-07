import torch
from final_model import ResNet1HeadID
from torchvision.models.feature_extraction import create_feature_extractor

def get_layer_names(model):
    layer_names = []
    for layer_name, _ in model.named_children():
        layer_names.append(layer_name)
    return layer_names

def flatten_features(outputs):
    for key, item in outputs.items():
        outputs[key] = item.flatten()
    return outputs

def make_prediction(model, flattened_dict, in_feat_model):
    preds = {}
    for key, item in flattened_dict.items():
        adj_layer = torch.nn.Linear(len(item), in_feat_model)
        adj_layer.requires_grad = False
        adj_output = adj_layer(item)
        pred = model.head(adj_output)
        preds[key] = pred
    return preds

if __name__ == "__main__":
    checkpoint_path = r"C:\Users\rvacher\Downloads\trained_model.pt"
    output_size = 100
    feature_extractor = None
    sub_module = False
    rand_image = torch.rand([1, 3, 224, 224])

    # Check if GPU is available and if not, use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load in model 
    trained_model_state_dict = torch.load(checkpoint_path, map_location = device)

    trained_model = ResNet1HeadID(output_size = output_size, feature_extractor= feature_extractor).eval()
    trained_model.load_state_dict(trained_model_state_dict["model_state_dict"])
    trained_model_backbone = trained_model.feature_extractor

    #Create feature extractor 
    output_layer_names = get_layer_names(trained_model_backbone)[:-1]
    feature_extractor = create_feature_extractor(trained_model_backbone, output_layer_names)

    #Extract features & predict fMRI data 
    outputs = feature_extractor(rand_image)
    flat_outputs = flatten_features(outputs)
    predictions = make_prediction(trained_model, flat_outputs, trained_model.head.in_features)
    print(predictions)

