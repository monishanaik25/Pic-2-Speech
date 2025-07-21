document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');

    form.onsubmit = async function(e) {
        e.preventDefault();
        
        loadingDiv.style.display = 'block';
        resultDiv.innerHTML = '';

        const formData = new FormData(form);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                resultDiv.innerHTML = `<div class="error">${data.error}</div>`;
            } else {
                let resultHTML = `
                    <div class="success">
                        <p><strong>Original Caption:</strong> ${data.caption}</p>
                `;
                
                if (data.translated_caption) {
                    resultHTML += `
                        <p><strong>Translated Caption:</strong> ${data.translated_caption}</p>
                    `;
                }
                
                resultHTML += '</div>';
                resultDiv.innerHTML = resultHTML;
            }
        } catch (error) {
            resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        } finally {
            loadingDiv.style.display = 'none';
        }
    };
});