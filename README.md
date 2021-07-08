# Latent space visualization via t-SNE
This code is used for latent space visualization in Image-to-Image problems. Firstly, all images are passed through ResNet101 to extract features, which are then reduced dimension by t-SNE algorithm for plotting

![Example result: latent space of Monet style and Photo style](https://github.com/manhkhanhad/latent-space-visualization-I2I/blob/main/images/Monet2Photo.png)
<h2>Table of content:</h2>
    <ol>
        <li><a href="#heading1">Installation</a></li>
        <li>
            <a href="#heading2">Basic Usage</a>
            <ol>
                <li><a href="#use_cmd">Commands line</li>
                <li><a href="#use_code">Python code</a>
                </li>
            </ol>
        </li>
    </ol>
  
  
 
<h2 id="heading1">Installation</h2>
<h2 id="heading2">Basic Usage</h2>
<h3 id="use_cmd">1. Commands line</h3>
To visualize latent space of n domain. All images of each domain are put into a folder.

```
python --path [path] --label [label] --path_save result_path --n_sample number_of_sample --n_components number_of_component --perplexity perplexity --n_iter
```

where `--path` is a list of path to domain, `--label` is a list name of domain
<h4>Example</h3>
We have two folders: one with Monet style images and the other with Photo style images. To visualize these two domain in a latent space we use command line

```
python --path path/to/Monet_folder path/to/Photo_folder --label Monet_style Photo --n_sample 500 --n_components 2 --perplexity 40 --n_iter 4000
```

<h3 id="use_code">2. Python code</h3>
