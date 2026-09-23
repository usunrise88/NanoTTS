import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import tseslint from 'typescript-eslint'

/** The consistency rules below are the point of this config. Everything a page
 *  author could do inconsistently — spacing, headings, overlays, motion,
 *  fetching — is either impossible or fails the build. */
export default tseslint.config(
  { ignores: ['dist', 'src/routeTree.gen.ts', 'node_modules'] },

  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,

  // The config file itself is plain JS and sits outside the app's program, so
  // the type-aware rules cannot run on it.
  { files: ['**/*.js'], ...tseslint.configs.disableTypeChecked },

  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: {
        project: ['./tsconfig.app.json', './tsconfig.node.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: { 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/consistent-type-imports': ['error', { fixStyle: 'inline-type-imports' }],
      '@typescript-eslint/no-misused-promises': ['error', { checksVoidReturn: false }],
    },
  },

  // --- motion is opt-in ----------------------------------------------------
  // Tailwind's animation utilities are banned outside the primitives. The few
  // transitions the app has live in ui/ and are switched off under
  // prefers-reduced-motion by the global stylesheet.
  {
    files: ['src/**/*.tsx'],
    ignores: ['src/components/ui/**'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: "Literal[value=/\\b(animate-|transition|duration-|ease-)\\S*/]",
          message:
            'Animations are off by default. If a surface genuinely needs motion, add it to a primitive in components/ui and make it reduced-motion aware.',
        },
      ],
    },
  },

  // --- pages are built from primitives, nothing else ------------------------
  {
    files: ['src/routes/**/*.tsx'],
    ignores: ['src/routes/__root.tsx'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: 'JSXOpeningElement[name.name=/^h[1-6]$/]',
          message:
            'Page headings come from the route\'s staticData and are rendered by the shell. Use Section for in-page labels.',
        },
        {
          selector: 'JSXOpeningElement[name.name=/^(button|input|textarea|select|table|dialog)$/]',
          message: 'Use the primitive from components/ui instead of a raw element.',
        },
        {
          selector: "Literal[value=/\\b(animate-|transition|duration-|ease-)\\S*/]",
          message: 'Animations are off by default; see components/ui.',
        },
        {
          selector: "Literal[value=/^(?=.*\\b(m|p)[trblxy]?-\\d)/]",
          message:
            'Spacing belongs to the layout. Compose Page, Section, Stack and Columns instead of setting margins or padding on a page.',
        },
      ],
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['@base-ui-components/*', 'sonner', '@radix-ui/react-*'],
              message:
                'Overlays and toasts go through components/ui so behaviour and motion stay identical everywhere.',
            },
          ],
        },
      ],
    },
  },

  // --- one way to reach the server -----------------------------------------
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/lib/api.ts'],
    rules: {
      'no-restricted-globals': [
        'error',
        { name: 'fetch', message: 'Call the server through lib/api so auth and errors stay uniform.' },
      ],
    },
  },

  {
    files: ['src/lib/notifications.tsx', 'src/components/ui/toaster.tsx'],
    rules: { 'no-restricted-imports': 'off' },
  },
)
