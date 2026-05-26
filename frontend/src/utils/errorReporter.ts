/**
 * Error Reporter - Frontend error logging and reporting utility
 * 
 * Captures and reports frontend errors to the backend for centralized monitoring
 */

interface ErrorReport {
  error_type: string;
  message: string;
  stack_trace?: string;
  severity: 'debug' | 'info' | 'warning' | 'error' | 'critical';
  url?: string;
  user_agent?: string;
  session_id?: string;
  user_id?: string;
  extra?: Record<string, unknown>;
}

interface ErrorReporterConfig {
  apiEndpoint: string;
  enabled: boolean;
  captureUnhandledRejections: boolean;
  captureGlobalErrors: boolean;
  batchSize: number;
  flushInterval: number;
}

class ErrorReporter {
  private config: ErrorReporterConfig;
  private errorQueue: ErrorReport[] = [];
  private flushTimer: ReturnType<typeof setInterval> | null = null;
  private sessionId: string;

  constructor(config?: Partial<ErrorReporterConfig>) {
    this.config = {
      apiEndpoint: '/api/v1/errors/report',
      enabled: true,
      captureUnhandledRejections: true,
      captureGlobalErrors: true,
      batchSize: 10,
      flushInterval: 5000,
      ...config,
    };

    this.sessionId = this.generateSessionId();

    if (this.config.enabled) {
      this.setupGlobalHandlers();
      this.startFlushTimer();
    }
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  private setupGlobalHandlers(): void {
    if (typeof window === 'undefined') return;

    if (this.config.captureGlobalErrors) {
      window.addEventListener('error', (event) => {
        this.report({
          error_type: event.error?.name || 'Error',
          message: event.message,
          stack_trace: event.error?.stack,
          severity: 'error',
          url: window.location.href,
          extra: {
            filename: event.filename,
            lineno: event.lineno,
            colno: event.colno,
          },
        });
      });
    }

    if (this.config.captureUnhandledRejections) {
      window.addEventListener('unhandledrejection', (event) => {
        const error = event.reason;
        this.report({
          error_type: error?.name || 'UnhandledRejection',
          message: error?.message || String(error),
          stack_trace: error?.stack,
          severity: 'error',
          url: window.location.href,
        });
      });
    }
  }

  private startFlushTimer(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.flushTimer = setInterval(() => {
      this.flush();
    }, this.config.flushInterval);
  }

  async report(error: ErrorReport): Promise<string | null> {
    if (!this.config.enabled) return null;

    const report: ErrorReport = {
      ...error,
      url: error.url || (typeof window !== 'undefined' ? window.location.href : undefined),
      user_agent: typeof navigator !== 'undefined' ? navigator.userAgent : undefined,
      session_id: this.sessionId,
    };

    this.errorQueue.push(report);

    if (this.errorQueue.length >= this.config.batchSize) {
      await this.flush();
    }

    return null;
  }

  async flush(): Promise<void> {
    if (this.errorQueue.length === 0) return;

    const errors = [...this.errorQueue];
    this.errorQueue = [];

    try {
      await Promise.all(
        errors.map((error) => this.sendError(error))
      );
    } catch (e) {
      console.error('[ErrorReporter] Failed to flush errors:', e);
      this.errorQueue.unshift(...errors);
    }
  }

  private async sendError(error: ErrorReport): Promise<void> {
    try {
      const response = await fetch(this.config.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(error),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (e) {
      console.error('[ErrorReporter] Failed to send error:', e);
      throw e;
    }
  }

  info(message: string, extra?: Record<string, unknown>): void {
    this.report({ error_type: 'Info', message, severity: 'info', extra });
  }

  warn(message: string, extra?: Record<string, unknown>): void {
    this.report({ error_type: 'Warning', message, severity: 'warning', extra });
  }

  error(message: string, error?: Error, extra?: Record<string, unknown>): void {
    this.report({
      error_type: error?.name || 'Error',
      message: error?.message || message,
      stack_trace: error?.stack,
      severity: 'error',
      extra,
    });
  }

  critical(message: string, error?: Error, extra?: Record<string, unknown>): void {
    this.report({
      error_type: error?.name || 'CriticalError',
      message: error?.message || message,
      stack_trace: error?.stack,
      severity: 'critical',
      extra,
    });
  }

  setUserId(userId: string): void {
    this.report({
      error_type: 'UserIdentification',
      message: `User identified: ${userId}`,
      severity: 'info',
      user_id: userId,
    });
  }

  destroy(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    this.flush();
  }
}

let reporterInstance: ErrorReporter | null = null;

export function initErrorReporter(config?: Partial<ErrorReporterConfig>): ErrorReporter {
  if (reporterInstance) {
    reporterInstance.destroy();
  }
  reporterInstance = new ErrorReporter(config);
  return reporterInstance;
}

export function getErrorReporter(): ErrorReporter {
  if (!reporterInstance) {
    return initErrorReporter();
  }
  return reporterInstance;
}

export { ErrorReporter };
