const grid = document.querySelector('[data-grid]');
const searchInput = document.querySelector('[data-search]');
const meta = document.querySelector('[data-meta]');
const emptyState = document.querySelector('[data-empty]');
const template = document.querySelector('#example-card');

const state = {
  examples: [],
  baseViewer: '../index.html',
};

const toQueryString = (args = {}) => {
  const params = new URLSearchParams();
  Object.entries(args).forEach(([key, value]) => {
    if (value === null || value === undefined) {
      return;
    }
    params.set(key, String(value));
  });
  return params.toString();
};

const buildViewerUrl = (args) => {
  const url = new URL(state.baseViewer, window.location.href);
  const query = toQueryString(args);
  if (query.length > 0) {
    url.search = query;
  }
  return url.toString();
};

const renderExamples = (examples) => {
  grid.innerHTML = '';

  if (examples.length === 0) {
    emptyState.hidden = false;
    return;
  }

  emptyState.hidden = true;

  const fragment = document.createDocumentFragment();
  examples.forEach((example) => {
    const card = template.content.cloneNode(true);
    const link = card.querySelector('.card-link');
    const title = card.querySelector('.example-title');
    const description = card.querySelector('.example-description');
    const id = card.querySelector('.card-id');
    const image = card.querySelector('img');
    const tagRow = card.querySelector('.tag-row');

    link.href = buildViewerUrl(example.args);
    title.textContent = example.title;
    description.textContent = example.description;
    id.textContent = example.id;
    image.src = example.thumbnail;
    image.alt = `${example.title} thumbnail`;

    if (Array.isArray(example.tags)) {
      example.tags.forEach((tag) => {
        const span = document.createElement('span');
        span.className = 'tag';
        span.textContent = tag;
        tagRow.appendChild(span);
      });
    }

    fragment.appendChild(card);
  });

  grid.appendChild(fragment);
};

const updateMeta = (total, filtered) => {
  meta.textContent = `${filtered} of ${total} examples`;
};

const filterExamples = (term) => {
  if (!term) {
    return state.examples;
  }

  const lowered = term.toLowerCase();
  return state.examples.filter((example) => {
    const text = [
      example.id,
      example.title,
      example.description,
      ...(example.tags || []),
    ]
      .join(' ')
      .toLowerCase();
    return text.includes(lowered);
  });
};

const loadExamples = async () => {
  const response = await fetch('./examples.json');
  if (!response.ok) {
    throw new Error(`Failed to load examples: ${response.status}`);
  }
  const data = await response.json();
  state.examples = data.examples || [];
  state.baseViewer = data.base_viewer || state.baseViewer;

  const filtered = filterExamples(searchInput.value.trim());
  renderExamples(filtered);
  updateMeta(state.examples.length, filtered.length);
};

searchInput.addEventListener('input', (event) => {
  const value = event.target.value.trim();
  const filtered = filterExamples(value);
  renderExamples(filtered);
  updateMeta(state.examples.length, filtered.length);
});

loadExamples().catch((error) => {
  console.error(error);
  meta.textContent = 'Failed to load examples.';
});
