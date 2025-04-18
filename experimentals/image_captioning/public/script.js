document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const generateBtn = document.getElementById('generateBtn');
    const loadingIndicator = document.getElementById('loading');
    const imageInput = document.getElementById('image');
    const generatedText = document.getElementById('generatedText');
    const output = document.getElementById('output');
    const preview = document.getElementById('preview');
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const changeImageBtn = document.getElementById('changeImageBtn');
    const copyBtn = document.getElementById('copyBtn');

    console.log('Script initialized');

    // ===== FILE HANDLING FUNCTIONS =====

    // Process the selected file (whether from browse or drop)
    function processFile(file) {
        // Hide any previous results when changing the image
        output.classList.add('hidden');

        if (!file) return;

        console.log('Processing file:', file.name, 'Type:', file.type, 'Size:', file.size);

        // Validate file type
        if (!file.type.match('image.*')) {
            alert('Please select an image file (JPEG, PNG, etc.)');
            return;
        }

        // Use FileReader API for reading the file
        const reader = new FileReader();

        // Set up the onload handler
        reader.onload = function(e) {
            // Set the preview image source
            preview.src = e.target.result;

            // When the image is loaded in the DOM
            preview.onload = function() {
                // Show the preview container and hide upload area
                uploadArea.classList.add('hidden');
                previewContainer.classList.remove('hidden');
                // Enable the generate button
                generateBtn.disabled = false;
            };
        };

        // Handle errors
        reader.onerror = function() {
            console.error('Error reading file');
            alert('Error reading the selected image. Please try another image.');
        };

        // Start reading the file as a data URL
        reader.readAsDataURL(file);
    }

    // ===== EVENT HANDLERS =====

    // Handle file input change (browse files)
    imageInput.addEventListener('change', function(e) {
        const file = this.files[0];
        if (file) {
            processFile(file);
        }
    });

    // Handle click on the upload area
    uploadArea.addEventListener('click', function() {
        // Clear the input to ensure change event fires even if selecting the same file
        imageInput.value = '';
        imageInput.click();
    });

    // Handle click on change image button
    changeImageBtn.addEventListener('click', function() {
        // Clear the input to ensure change event fires even if selecting the same file
        imageInput.value = '';
        imageInput.click();
    });

    // ===== DRAG AND DROP HANDLERS =====

    // Highlight drop area when file is dragged over
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.add('dragover');
        }, false);
    });

    // Remove highlight when file is dragged out
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();

        const file = e.dataTransfer.files[0];
        if (file) {
            processFile(file);
        }
    });

    // Prevent default behavior for drag events on the document
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.addEventListener(eventName, function(e) {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    // ===== COPY BUTTON =====

    copyBtn.addEventListener('click', () => {
        const textToCopy = generatedText.textContent;
        navigator.clipboard.writeText(textToCopy).then(() => {
            copyBtn.classList.add('copied');

            // Show feedback
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            `;

            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
                copyBtn.classList.remove('copied');
            }, 2000);
        });
    });

    // ===== GENERATE CAPTION =====

    generateBtn.addEventListener('click', async () => {
        try {
            output.classList.add('hidden');
            loadingIndicator.classList.remove('hidden');
            generateBtn.disabled = true;

            const formData = new FormData();
            if (imageInput.files[0]) {
                formData.append('image', imageInput.files[0]);
            }

            // // Simulate server delay
            // await new Promise(resolve => setTimeout(resolve, 1000));

            // // Use fake data instead of server request
            // const data = {
            //     answer: "A beautiful scenic landscape with mountains in the background and a serene lake reflecting the sky."
            // };
            const response = await fetch('http://172.16.20.54:8006/generate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            generatedText.textContent = data.answer;
            output.classList.remove('hidden');
            // Scroll to result
            output.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            console.error('Error:', error);
            showError('An error occurred while captioning the image. Please try again.');
        } finally {
            loadingIndicator.classList.add('hidden');
            generateBtn.disabled = false;
        }
    });

    // ===== ERROR HANDLING =====

    function showError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = message;
        errorElement.style.backgroundColor = '#fee2e2';
        errorElement.style.color = '#ef4444';
        errorElement.style.padding = '12px';
        errorElement.style.borderRadius = '8px';
        errorElement.style.marginTop = '20px';
        errorElement.style.textAlign = 'center';

        // Remove any existing error
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        // Add to DOM
        generateBtn.insertAdjacentElement('afterend', errorElement);

        // Auto remove after 5 seconds
        setTimeout(() => {
            errorElement.style.opacity = '0';
            errorElement.style.transition = 'opacity 0.5s ease';
            setTimeout(() => errorElement.remove(), 500);
        }, 5000);
    }
});
