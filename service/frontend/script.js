/**
 * Minimal client logic for the ATRIUM LLM Enrichment demo frontend.
 * Sends pasted lines to POST /extract_keywords_text, or an uploaded file to
 * POST /extract_keywords, and renders the per-line keyword table.
 */
function renderResults(data) {
    const container = document.getElementById('results');
    if (!data || !Array.isArray(data.lines)) {
        container.innerHTML = '<div class="error">Unexpected response format from the API.</div>';
        return;
    }

    let html = `<h3>Results for ${data.doc_id} (backend: ${data.backend}, model: ${data.model || 'n/a'})</h3>`;
    const stats = data.stats || {};
    html += `<p>processed: ${stats.processed ?? '?'} · filtered out: ${stats.skipped_filter ?? '?'} · errors: ${stats.skipped_error ?? '?'}</p>`;
    html += '<table border="1" cellpadding="6" style="border-collapse: collapse; width: 100%;">';
    html += '<tr><th>P/L</th><th>Text</th><th>Keywords (cs)</th><th>Keywords (en)</th><th>Category</th><th>Conf.</th></tr>';
    data.lines.forEach(line => {
        html += `<tr>
            <td>${line.page ?? ''}/${line.line ?? ''}</td>
            <td>${line.text ?? ''}</td>
            <td>${(line.keywords_cs || []).join(', ')}</td>
            <td>${(line.keywords_en || []).join(', ')}</td>
            <td>${line.category ?? ''}</td>
            <td>${line.confidence != null ? line.confidence.toFixed(2) : ''}</td>
        </tr>`;
    });
    html += '</table>';
    container.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('extractForm');
    const loader = document.getElementById('loader');
    const resultDiv = document.getElementById('results');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        resultDiv.innerHTML = '';
        loader.style.display = 'block';

        const lines = document.getElementById('linesInput').value
            .split('\n').map(s => s.trim()).filter(Boolean);
        const file = document.getElementById('fileInput').files[0];
        const backend = document.getElementById('backendInput').value;
        const topK = parseInt(document.getElementById('topkInput').value, 10) || 10;

        const baseUrl = window.location.origin.includes('localhost') ? 'http://localhost:8000' : '';

        try {
            let response;
            if (file) {
                const formData = new FormData();
                formData.append('file', file);
                if (backend) formData.append('backend', backend);
                formData.append('top_k', topK);
                response = await fetch(`${baseUrl}/extract_keywords`, { method: 'POST', body: formData });
            } else if (lines.length) {
                const payload = { lines: lines, top_k: topK };
                if (backend) payload.backend = backend;
                response = await fetch(`${baseUrl}/extract_keywords_text`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            } else {
                throw new Error('Paste some text lines or choose a file first.');
            }

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail ? JSON.stringify(data.detail) : `Server error: ${response.status}`);
            }
            renderResults(data);
        } catch (err) {
            console.error(err);
            resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${err.message}</div>`;
        } finally {
            loader.style.display = 'none';
        }
    });
});
