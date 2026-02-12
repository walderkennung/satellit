const fs = require('fs');
const path = require('path');

const contentDir = path.join(__dirname, 'content');
const apiModulesDir = path.join(contentDir, 'api', 'satellit_sam');

function getApiNavigationChildren() {
  const children = [{ title: 'Python API Overview', path: 'api', icon: 'file-text' }];

  if (!fs.existsSync(apiModulesDir)) {
    return children;
  }

  const files = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        files.push(fullPath);
      }
    }
  };

  walk(apiModulesDir);
  files.sort();

  for (const filePath of files) {
    const contentRelativePath = path
      .relative(contentDir, filePath)
      .replace(/\\/g, '/');
    const pathWithoutExtension = contentRelativePath.replace(/\.md$/, '');
    const title = pathWithoutExtension
      .replace(/^api\//, '')
      .replace(/\//g, '.');
    children.push({ title, path: pathWithoutExtension, icon: 'box' });
  }

  return children;
}

const apiNavigationChildren = getApiNavigationChildren();

// docmd.config.js
module.exports = {
  // --- Core Metadata ---
  siteTitle: 'Walderkennung Satellit Documentation',
  siteUrl: '', // e.g. https://mysite.com (Critical for SEO/Sitemap)

  // --- Branding ---
  logo: {
    light: 'assets/images/docmd-logo-dark.png',
    dark: 'assets/images/docmd-logo-light.png',
    alt: 'Logo',
    href: './',
  },
  favicon: 'assets/favicon.ico',

  // --- Source & Output ---
  srcDir: 'content',
  outputDir: 'site',

  // --- Theme & Layout ---
  theme: {
    name: 'sky',            // Options: 'default', 'sky', 'ruby', 'retro'
    defaultMode: 'system',  // 'light', 'dark', or 'system'
    enableModeToggle: true, // Show mode toggle button
    positionMode: 'top',    // 'top' or 'bottom'
    codeHighlight: true,    // Enable Highlight.js
    customCss: [],          // e.g. ['assets/css/custom.css']
  },

  // --- Features ---
  search: true,           // Built-in offline search
  minify: true,           // Minify HTML/CSS/JS in build
  autoTitleFromH1: true,  // Auto-generate page title from first H1
  copyCode: true,         // Show "copy" button on code blocks
  pageNavigation: true,   // Prev/Next buttons at bottom

  // --- Navigation (Sidebar) ---
  navigation: [
    { title: 'Introduction', path: '/', icon: 'home' },
    {
      title: 'CLI',
      icon: 'terminal',
      collapsible: true,
      children: [
        { title: 'CLI Application', path: 'cli/application', icon: 'terminal-square' },
      ],
    },
    {
      title: 'Knowledge Base',
      icon: 'book-open',
      collapsible: true,
      children: [
        { title: 'SAM3 Transfer Learning Overview', path: 'knowledgebase/fine_tuning_transfer_learning', icon: 'map' },
        { title: 'Goals and Scope', path: 'knowledgebase/fine_tuning_transfer_learning/goals_scope', icon: 'target' },
        { title: 'Transfer Learning Fundamentals', path: 'knowledgebase/fine_tuning_transfer_learning/fundamentals', icon: 'book-text' },
        { title: 'Phased Strategy', path: 'knowledgebase/fine_tuning_transfer_learning/phased_strategy', icon: 'route' },
        { title: 'Transfer Learning Methods', path: 'knowledgebase/fine_tuning_transfer_learning/adaptation_methods', icon: 'layers' },
        { title: 'SAM Architecture Variants', path: 'knowledgebase/fine_tuning_transfer_learning/sam_architecture_variants', icon: 'cpu' },
        { title: 'Data Requirements', path: 'knowledgebase/fine_tuning_transfer_learning/data_requirements', icon: 'database' },
        { title: 'Implementation Plan', path: 'knowledgebase/fine_tuning_transfer_learning/implementation_plan', icon: 'hammer' },
        { title: 'Experiment Roadmap', path: 'knowledgebase/fine_tuning_transfer_learning/experiment_roadmap', icon: 'flask' },
        { title: 'Evaluation and Decision Matrix', path: 'knowledgebase/fine_tuning_transfer_learning/evaluation_decision', icon: 'check-circle' },
      ],
    },
    {
      title: 'API Reference',
      icon: 'code',
      collapsible: true,
      children: apiNavigationChildren,
    },
    { title: 'GitHub', path: 'https://github.com/walderkennung/satellit', icon: 'github', external: true },
  ],

  // --- Plugins ---
  plugins: {
    seo: {
      defaultDescription: 'Documentation built with docmd.',
      openGraph: {
        defaultImage: '',   // e.g. 'assets/images/og-image.png'
      },
      twitter: {
        cardType: 'summary_large_image',
      }
    },
    analytics: {
      googleV4: {
        measurementId: 'G-X9WTDL262N' // Replace with your Google Analytics Measurement ID
      }
    },
    sitemap: {
      defaultChangefreq: 'weekly',  // e.g. 'daily', 'weekly', 'monthly'
      defaultPriority: 0.8          // Priority between 0.0 and 1.0
    }
  },

  // --- Footer ---
  footer: '© ' + new Date().getFullYear() + ' Walderkennung Satellit. Built with [docmd](https://docmd.io).',

  // --- Edit Link ---
  editLink: {
    enabled: false,
    baseUrl: 'https://github.com/walderkennung/satellit/edit/main/docs/content',
    text: 'Edit this page'
  }
};
