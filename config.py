cmd = {
       	"--data": "/mnt/beegfs/home/devatine/exp/code/LearningPermutations/data/",
        "--embeddings": "/mnt/beegfs/home/devatine/exp/code/LearningPermutations/embeddings/",
        "--train-langs": "en_gum",
        "--dev-langs": "en_gum,fr_ftb,de_gsd",
        "--device": "cuda",
        "--epochs": "10",
        "--samples": "1000",
}

cmd_options = [
[
        ("lr_1e-3", {"--lr": "0.001"}),
        ("lr_5e-3", {"--lr": "0.005"}),
],
[
        ("proj_dim128", {"--proj-dim": "128"}),
        ("proj_dim256", {"--proj-dim": "256"}),
        ("proj_dim512", {"--proj-dim": "512"}),
        ("proj_dim1024", {"--proj-dim": "1024"}),
],
[
        ("decayrate090", {"--decay-rate": "0.90"}),
        ("decayrate097", {"--decay-rate": "0.97"}),
        ("decayrate099", {"--decay-rate": "0.99"}),
],
]


