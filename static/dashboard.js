// Keep text opaque while adjusting only the two kinds of surface backgrounds.
function opacityControl(id, outputId, property, minimum, initial, storageKey) {
  const slider = document.getElementById(id);
  const output = document.getElementById(outputId);
  if (!slider || !output) return;
  function apply(value) {
    const number = Number(value);
    const opacity = Number.isFinite(number) ? Math.min(100, Math.max(minimum, number)) : initial;
    document.documentElement.style.setProperty(property, opacity / 100);
    slider.value = opacity;
    output.value = `${opacity}%`;
    output.textContent = `${opacity}%`;
    try { localStorage.setItem(storageKey, String(opacity)); } catch (_) { /* Private storage may be disabled. */ }
  }
  let saved;
  try { saved = localStorage.getItem(storageKey); } catch (_) { saved = null; }
  apply(saved === null ? initial : saved);
  slider.addEventListener('input', () => apply(slider.value));
}
opacityControl('opacity', 'opacity-value', '--panel-opacity', 70, 94, 'routewise-panel-opacity');
opacityControl('chart-opacity', 'chart-opacity-value', '--chart-opacity', 80, 100, 'routewise-chart-opacity');

const plotConfig = {
  responsive: true,
  displaylogo: false,
  displayModeBar: true,
  scrollZoom: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  toImageButtonOptions: { format: 'png', filename: 'delivery-chart', scale: 2 },
};
const modal = document.getElementById('chart-modal');
const expandedChart = document.getElementById('expanded-chart');
const modalTitle = document.getElementById('chart-modal-title');

function expandGraph(element) {
  if (!modal || !window.Plotly || !element.data) return;
  modalTitle.textContent = element.closest('.chart-card').querySelector('h3').textContent;
  modal.showModal();
  const layout = JSON.parse(JSON.stringify(element.layout));
  delete layout.width;
  delete layout.height;
  layout.autosize = true;
  requestAnimationFrame(() => Plotly.newPlot(expandedChart, element.data, layout, plotConfig));
}
document.getElementById('close-chart')?.addEventListener('click', () => modal.close());
modal?.addEventListener('close', () => { if (window.Plotly) Plotly.purge(expandedChart); });

async function loadGraph(element) {
  try {
    if (!window.Plotly) throw new Error('Chart library is unavailable.');
    const response = await fetch(element.dataset.chartUrl);
    if (!response.ok) throw new Error('Chart data could not be loaded.');
    const specification = await response.json();
    element.replaceChildren();
    await Plotly.newPlot(element, specification.data, specification.layout, plotConfig);
    const button = document.createElement('button');
    button.className = 'expand-chart';
    button.type = 'button';
    button.textContent = 'Expand ⤢';
    button.setAttribute('aria-label', `Expand ${element.getAttribute('aria-label')}`);
    button.addEventListener('click', () => expandGraph(element));
    element.closest('.chart-card').querySelector('.chart-heading').append(button);
  } catch (error) {
    element.textContent = `${error.message} Refresh to retry.`;
    element.classList.add('chart-error');
  }
}
// Load near-visible graphs first; keep the initial page responsive on a student laptop.
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        observer.unobserve(entry.target);
        loadGraph(entry.target);
      }
    });
  }, { rootMargin: '300px' });
  document.querySelectorAll('.interactive-chart').forEach(element => observer.observe(element));
} else {
  document.querySelectorAll('.interactive-chart').forEach(loadGraph);
}
