document.getElementById("generateBtn").addEventListener("click", async () => {
    const prompt = document.getElementById("prompt").value.trim();
    const negativePrompt = document.getElementById("negativePrompt").value.trim();
    const imageInput = document.getElementById("image");
    const loading = document.getElementById("loading");
    const imageContainer = document.getElementById("imageContainer");

    const seed = parseInt(document.getElementById("seed").value) || 42;
    const guidanceScale = parseFloat(document.getElementById("guidance").value) || 7.5;
    const numInferenceSteps = parseInt(document.getElementById("steps").value) || 30;
    const numVideosPerPrompt = parseInt(document.getElementById("numberofVideos").value) || 1;
    const numFramesPerVideo = parseInt(document.getElementById("numFrames").value) || 1;
    const fps = parseInt(document.getElementById("fps").value) || 1;

    if (!prompt) {
        alert("Please enter a prompt!");
        return;
    }

    if (!imageInput.files || imageInput.files.length === 0) {
        alert("Please upload an image!");
        return;
    }

    const imageFile = imageInput.files[0];

    loading.classList.remove("hidden");
    imageContainer.classList.add("hidden");
    imageContainer.innerHTML = "";

    const formData = new FormData();
    const requestData = {
        prompt,
        negative_prompt: negativePrompt || null,
        num_inference_steps: numInferenceSteps,
        guidance_scale: guidanceScale,
        seed,
        nums_frames: numFramesPerVideo,
        fps
    };

    formData.append("request_data", JSON.stringify(requestData));
    formData.append("image", imageFile);

    try {
        const response = await fetch("http://172.16.20.54:8000/generate", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error("Failed to generate video");

        const data = await response.json();
        const videoBase64 = data.image;

        if (!videoBase64) {
            alert("No video generated!");
            return;
        }
        for (const base64 of videoBase64) {
            const video = document.createElement("video");
            console.log(base64);
            video.controls = true;
            video.src = `data:video/mp4;base64,${base64}`;
            video.classList.add("generated-video");
            imageContainer.appendChild(video);
        }
        imageContainer.classList.remove("hidden");
    } catch (error) {
        console.error("Error:", error);
        alert("Error generating video!");
    } finally {
        loading.classList.add("hidden");
    }
});
