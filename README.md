# latent-upscale
### An InvokeAI Node to Upscale Latents using a Trained Model

Ported from [city96's SD-Latent-Upscaler](https://github.com/city96/SD-Latent-Upscaler) to an [InvokeAI](https://github.com/invoke-ai/invokeai/) node for the workflow editor.

This node uses a small (~2.4mb) model to upscale the latents used in a Stable Diffusion 1.5 or Stable Diffusion XL image generation, rather than the typical interpolation method.

## Installation

Navigate to your InvokeAI installation's `nodes` folder, and run
```
git clone https://github.com/gogurtenjoyer/latent-upscale
```
or, just click the green button in the upper-right here, download the zip, and unzip it in your `nodes` folder.

If you'd like, you can also manually download models from the original author's HF repo [here](https://huggingface.co/city96/SD-Latent-Upscaler/tree/main), or just let the node auto-download them on-demand (they are very small, so it'll be quick). If you do choose to download them yourself, make sure you grab the latest (v2.1) version of the models, make a new `models` folder within this node's folder, and place the safetensor files there.

## Usage

Example workflows for SD 1.5 and SDXL are included. You can load these in the workflow editor to use this quickly, but otherwise you'd just be replacing InvokeAI's built-in Scale Latents or Resize Latents node with this one in an existing workflow. In general, the input would be the latents from Denoise Latents, and the output would go to another Denoise Latents for the high res portion of denoising. You'd also send the new upscaled width and height to the Noise node feeding the high res Denoise Latents.

The settings are pretty simple:
- Latent Ver: The version of Stable Diffusion you're using in your workflow. `v1` for SD 1.5, and `XL` for SDXL.
- Scale Factor: Unlike the other nodes dealing with latents, you are limited to 1.25, 1.5, and 2.0 scaling. If you'd like another ratio, chain two (or more) nodes together! Get creative with terrible, terrible math.
- Magic Number: That's cute, but I have [no](https://discuss.huggingface.co/t/what-does-0-18215-mean-in-blog-stable-diffusion-with-diffusers/24993) idea what you're talking about. If there were such a number, changing it would probably be painful.

The combination of the above two settings will choose the proper local model, or download it from Hugging Face. The node runs VERY quickly, so don't worry if you'd like to chain them together for another scale factor.

## Why?
Latent upscale had been a popular way to create images larger than the original model was trained for. It had a lot of upsides and could create very sharp, textured, and detailed larger images. However, simply interpolating the latents to a larger size could also cause really obvious artifacting in some images. Months ago, I had moved from latent upscale to our more 'modern' method, but wasn't totally happy with it, either. 

Last week, I was going through my older images made with latent upscale and I was amazed by the detail and sharpness (aside from the 10-20% of images which had the low res latent artifacts). I decided that there must still be a benefit to latent upscale, if the problematic aspects could be solved. I had heard of another latent upscale model, made by people from HF and StabilityAI, but this model was 1.5gb. A fellow InvokeAI dev/node dev pointed me to city96's code and models, and here we are: that crisp textured detailed look I love, without the nasty artifacting.

## Comparison with Traditional Latent Upscale
I chose images with 'text' because it was an easy way to get comparison examples, but this problem crops up in any areas of stark shapes/contrast (for example, chrome trim on a classic car, or a window frame).

On the left: traditional latent upscale. On the right: this node's output. All other settings are the same.

![a side-by-side comparison of two latent upscaled images using different methods](/comparison.png?raw=true)

![a side-by-side comparison of two latent upscaled images using different methods](/comparison2.png?raw=true)
