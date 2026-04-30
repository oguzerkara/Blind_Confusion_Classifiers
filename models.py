# ----------------------------------------------------------
# ---------------       Model Loading      -----------------
# ----------------------------------------------------------

import torch
import torchvision.models as models
import timm

def load_model(name, DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    n = name.lower()
    #### Traditional [1-2] ####
    # AlexNet
    if n == "alexnet":
        weights = models.AlexNet_Weights.IMAGENET1K_V1
        model = models.alexnet(weights=weights)

    # VGG16
    elif n == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)

    #### ResNet Based [3-11] ####
    elif n == "resnet200d":
        model = timm.create_model("resnet200d", pretrained=True)
        weights = None
    elif n == "resnet101" :
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        model = models.resnet101(weights=weights)
    elif n == "resnet50" :
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    # ResNeXt, WideResNet
    elif n == "resnext101":
        weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
        model   = models.resnext101_32x8d(weights=weights)
    elif n == "wideresnet101_2":
        weights = models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
        model   = models.wide_resnet101_2(weights=weights)

    # Inception‐ResNet‐v2
    elif n == "inception_resnet_v2":
        model = timm.create_model('inception_resnet_v2', pretrained=True)
        weights = None
    # Inception - GoogleNet
    elif n == "inception_v3" :
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights = weights)

    # MobileNet
    elif n == "mobilenet_v2" :
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights = weights)

    # DenseNets
    elif n == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model   = models.densenet121(weights=weights)
    elif n == "densenet161":
        weights = models.DenseNet161_Weights.IMAGENET1K_V1
        model   = models.densenet161(weights=weights)
    elif n == "densenet201":
        weights = models.DenseNet201_Weights.IMAGENET1K_V1
        model   = models.densenet201(weights=weights)


    #### Advanced [12-22] ####
    # ConvNeXts 
    elif n == "convnext_xlarge":
        model = timm.create_model("convnext_xlarge", pretrained=True)
        weights = None
    elif n == "convnext_large":
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    elif n == "convnext_base":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    elif n == "convnextv2_large":
        model = timm.create_model('convnextv2_large', pretrained=True)
        weights = None
    elif n == "convnextv2_base":
        model = timm.create_model('convnextv2_base', pretrained=True)
        weights = None
    # EfficientNets
    elif n == "efficientnet_v2_l":
        weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        model   = models.efficientnet_v2_l(weights=weights)
    elif n == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model   = models.efficientnet_v2_s(weights=weights)
    
    elif n == "efficientnet_b7":
        weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b7(weights=weights)
    elif n == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b3(weights=weights)
    elif n == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b4(weights=weights)
    elif n == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b0(weights=weights)
    # NFNets
    elif n == "nfnet_f5":
        model = timm.create_model('dm_nfnet_f5', pretrained=True)
        weights = None
    elif n == "nfnet_f4":
        model = timm.create_model('dm_nfnet_f4', pretrained=True)
        weights = None
    elif n == "nfnet_f2":
        model = timm.create_model('dm_nfnet_f2', pretrained=True)
        weights = None
    # ResNeSt adds split‑attention blocks for robustness
    elif n == "resnest50d":       
        weights = None
        model   = timm.create_model('resnest50d', pretrained=True)
    elif n == "resnest101e":
        weights = None
        model   = timm.create_model('resnest101e', pretrained=True)

    #### Robustly Trained [23-26] ####
    # NoisyStudent
    elif n == "noisystudent_b0":
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
        weights = None
    elif n == "noisystudent_b3":
        model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True)
        weights = None
    elif n == "noisystudent_b4":
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
        weights = None
    elif n == "noisystudent_b7":
        model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
        weights = None
    elif n == "noisystudent_v2_l":  # alias olarak böyle çağırmak istersen
        model = timm.create_model('tf_efficientnetv2_l_in21ft1k', pretrained=True)
        weights = None


    # AugMix ResNet-50
    elif n == "augmix":
        model = timm.create_model('resnet50.ram_in1k', pretrained=True)
        weights = None

    # BiT-M & BiT-L (ResNet-V2)
    elif n == "bit_m":
            # BiT-M uses a ResNet-V2-101×1 from Timm
        model   = timm.create_model('resnetv2_101x1_bitm', pretrained=True)
        weights = None

    ####  Vision Transformers [27-30] ####
    # Vanilla ViTs
    elif n == "vit_large16":
        model = timm.create_model('vit_large_patch16_224', pretrained=True)
        weights = None
    elif n == "vit_base16":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        weights = None

    # Swin Transformers
    elif n == "swin_base":
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        weights = None
    elif n == "swin_large":
        model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
        weights = None
    # DeiT
    elif n == "deit_base":
        model = timm.create_model('deit_base_patch16_224', pretrained=True)
        weights = None
    elif n == "maxvit_base":
        model = timm.create_model('maxvit_base_tf_224', pretrained=True)
        weights = None

    else:
        raise ValueError()

    # torchvision pipeline
    if weights is not None:
        preprocess = weights.transforms()        
        mean = torch.tensor(preprocess.mean).view(3,1,1)
        std  = torch.tensor(preprocess.std).view(3,1,1)
        input_size = preprocess.crop_size if hasattr(preprocess, 'crop_size') else preprocess.resize[1]
       
    # timm pipeline
    else:
        config  = model.default_cfg
        mean = torch.tensor(config['mean']).view(3,1,1)
        std  = torch.tensor(config['std']).view(3,1,1)
        input_size = config['input_size'][1]

    mean = torch.tensor(mean, device=DEVICE).view(3,1,1)
    std  = torch.tensor(std,  device=DEVICE).view(3,1,1)
    model = model.to(DEVICE).eval()

    mean = mean.to(DEVICE)
    std  = std.to(DEVICE)
    return model, (mean, std, input_size)
