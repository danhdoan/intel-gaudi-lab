document.getElementById("generateBtn").addEventListener("click", async () => {
    const prompt = document.getElementById("prompt").value.trim();
    const negativePrompt = document.getElementById("negativePrompt").value.trim();
    const imageInput = document.getElementById("image");
    const loading = document.getElementById("loading");

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

    const formData = new FormData();
    const requestData = {
        prompt,
        negative_prompt: negativePrompt || null,
        num_inference_steps: numInferenceSteps,
        guidance_scale: guidanceScale,
        seed,
        nums_frames: numFramesPerVideo,
        fps,
        num_video_per_prompt: numVideosPerPrompt,
    };

    formData.append("request_data", JSON.stringify(requestData));
    formData.append("image", imageFile);

    try {
        const response = await fetch("http://172.16.20.54:8008/generate", {
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
        for (let i = 0; i < videoBase64.length; i++) {
            const base64 = videoBase64[i];
            // Create a download link for each video
            const downloadLink = document.createElement("a");
            downloadLink.href = `data:video/mp4;base64,${base64}`;
            downloadLink.download = `video_${i + 1}.mp4`; // Naming each video
            downloadLink.style.display = 'none'; // Hide the link
            document.body.appendChild(downloadLink); // Append to body (required for click)
            downloadLink.click(); // Programmatically click the link to trigger download
            document.body.removeChild(downloadLink); // Remove the link after download
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Error generating video!");
    } finally {
        loading.classList.add("hidden");
    }
});
