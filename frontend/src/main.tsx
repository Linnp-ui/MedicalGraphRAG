import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import {initErrorReporter} from './utils/errorReporter';

initErrorReporter({
  enabled: true,
  captureGlobalErrors: true,
  captureUnhandledRejections: true,
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
