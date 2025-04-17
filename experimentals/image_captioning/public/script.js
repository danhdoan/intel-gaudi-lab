document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generateBtn');
    const loadingIndicator = document.getElementById('loading');
    const imageInput = document.getElementById('image');
    const generatedText = document.getElementById('generatedText');
    const output = document.getElementById('output');
    const preview = document.getElementById('preview');

    generateBtn.addEventListener('click', async () => {
        try {
            output.classList.add('hidden');
            loadingIndicator.classList.remove('hidden');
            generateBtn.disabled = true;

            const formData = new FormData();
            if (imageInput.files[0]) {
                formData.append('image', imageInput.files[0]);
            }

            const response = await fetch('http://172.16.20.54:8006/generate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            generatedText.innerHTML = `<p>${data.answer}</p>`;
            output.classList.remove('hidden');
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while captioning the image. Please try again.');
        } finally {
            loadingIndicator.classList.add('hidden');
            generateBtn.disabled = false;
        }
    });

    imageInput.addEventListener('change', () => {
        const file = imageInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.classList.remove('hidden');
                preview.classList.add('loaded');
                output.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        } else {
            preview.classList.add('hidden');
            preview.src = '';
        }
    });
});
