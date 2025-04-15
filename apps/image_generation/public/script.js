document.getElementById("generateBtn").addEventListener("click", async () => {
    const prompt = document.getElementById("prompt").value.trim();
    const negativePrompt = document.getElementById("negativePrompt").value.trim();
    const loading = document.getElementById("loading");
    const imageContainer = document.getElementById("imageContainer");

    const seed = parseInt(document.getElementById("seed").value) || 42;
    const guidanceScale = parseFloat(document.getElementById("guidance").value) || 7.5;
    const numInferenceSteps = parseInt(document.getElementById("steps").value) || 50;
    const batchSize = parseInt(document.getElementById("batchSize").value) || 1;
    const numImagesPerPrompt = parseInt(document.getElementById("numberImages").value) || 1;
    const size = parseInt(document.getElementById("size").value) || 512;

    if (!prompt) {
        alert("Please enter a prompt!");
        return;
    }
    if (batchSize > numImagesPerPrompt){
        alert("Batch size cannot be greater than number of images per prompt");
        return;
    }

    loading.classList.remove("hidden");
    imageContainer.classList.add("hidden");
    imageContainer.innerHTML = "";

    try {
        const response = await fetch("http://172.16.20.54:8002/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt,
                negative_prompt: negativePrompt || null,
                num_inference_steps: numInferenceSteps,
                num_images_per_prompt: numImagesPerPrompt,
                width: size,
                height: size,
                batch_size: batchSize,
                guidance_scale: guidanceScale,
                seed,
            }),
        });

        if (!response.ok) throw new Error("Failed to generate image");

        const data = await response.json();
        const images = data.image;

        if (!images || images.length === 0) {
            alert("No images generated!");
            return;
        }
        images.forEach((base64, i) => {
            const img = document.createElement("img");
            img.src = `data:image/png;base64,${base64}`;
            img.alt = `Generated Image ${i + 1}`;
            img.classList.add("generated-image");

            img.onload = () => img.classList.add("loaded");
            imageContainer.appendChild(img);
        });

        imageContainer.classList.remove("hidden");
    } catch (error) {
        console.error("Error:", error);
        alert("Error generating image!");
    } finally {
        loading.classList.add("hidden");
    }
});
