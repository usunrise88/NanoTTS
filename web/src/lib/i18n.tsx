import { createContext, use, useCallback, useMemo, useState, type ReactNode } from 'react'
import { safeLocal, storeLocal } from './utils'

export const LOCALES = ['ru', 'en'] as const
export type Locale = (typeof LOCALES)[number]

const dictionary = {
  ru: {
    'nav.synth': 'Синтез',
    'nav.models': 'Модели',

    'models.title': 'Модели',
    'models.description': 'Что установлено, что можно скачать и чем сейчас синтезируем.',
    'models.installed': 'Установленные',
    'models.available': 'Можно скачать',
    'models.available.hint':
      'Скачиваются модели, выпущенные сразу графами ONNX. Чекпоинт Pocket TTS так поставить нельзя: его нужно трассировать из весов, а для этого нужен Python из install.sh.',
    'models.allInstalled': 'Всё из каталога уже установлено.',
    'models.jobs': 'Загрузки',
    'models.active': 'активна',
    'models.activate': 'Переключить',
    'models.install': 'Скачать',
    'models.installing': 'Загрузка началась',
    'models.switched': 'Модель переключена',
    'models.removed': 'Модель удалена',
    'models.cancel': 'Отменить',
    'models.cancelled': 'Загрузка отменена',
    'models.col.model': 'Модель',
    'models.col.arch': 'Архитектура',
    'models.col.rate': 'Частота',
    'models.col.voices': 'Голоса',
    'models.col.size': 'Размер',
    'models.col.langs': 'Языки',
    'models.job.running': 'идёт',
    'models.job.done': 'готово',
    'models.job.failed': 'ошибка',
    'models.job.cancelled': 'отменена',
    'models.disabled.title': 'Реестр выключен',
    'models.disabled.body':
      'Сервер запущен без --models, поэтому ставить и переключать модели на ходу он не может.',

    'synth.guidance': 'Guidance',
    'synth.durationScale': 'Темп',
    'synth.archNote.pocket':
      'Авторегрессия: звук идёт с первого кадра, задержка не зависит от длины текста.',
    'synth.archNote.s3':
      'Сэмплер решает фразу целиком, поэтому задержка до первого звука растёт с длиной текста.',

    'voices.cloneUnsupported':
      'Эта модель поставляется с готовыми стилевыми векторами и не умеет выучивать голос с записи.',

    'nav.voices': 'Голоса',
    'nav.monitor': 'Мониторинг',
    'app.subtitle': 'Синтез русской речи на CPU',

    'conn.online': 'на связи',
    'conn.offline': 'нет связи',
    'conn.auth': 'нужен ключ',
    'conn.checking': 'проверяю',

    'key.label': 'API-ключ',
    'key.placeholder': 'Bearer-ключ, если сервис за ним',
    'key.hint': 'Хранится только в этом браузере и уходит заголовком Authorization.',
    'key.save': 'Сохранить',

    'synth.title': 'Синтез',
    'synth.description': 'Задержка до первого кадра и отношение времени синтеза к длительности.',
    'synth.text': 'Текст',
    'synth.textHint': 'Ударение: плюс перед гласной или U+0301 после неё',
    'synth.voice': 'Голос',
    'synth.format': 'Формат',
    'synth.temperature': 'Температура',
    'synth.eos': 'Порог EOS',
    'synth.seed': 'Seed',
    'synth.firstChunk': 'Первый чанк, кадров',
    'synth.run': 'Синтезировать',
    'synth.running': 'Синтезирую',
    'synth.request': 'Готовый запрос',
    'synth.copy': 'Скопировать',
    'synth.copied': 'Скопировано',
    'synth.pcmNote': 'Сырой PCM получен. Браузер его не проиграет, задержка измерена.',

    'metric.ttfb': 'TTFB',
    'metric.ttfbSub': 'до первого кадра',
    'metric.total': 'Всего',
    'metric.totalSub': 'wall clock',
    'metric.audio': 'Аудио',
    'metric.rtf': 'RTF',
    'metric.frames': 'кадров',
    'metric.realtime': 'реальное время',

    'voices.title': 'Голоса',
    'voices.description': 'Прогретые состояния FlowLM. Подставляются за миллисекунды вместо префилла.',
    'voices.warmTitle': 'Прогрев нового голоса',
    'voices.warmHint':
      'Референс идёт через Mimi и префилл FlowLM, результат — KV-состояние в safetensors. Моно или стерео, любая частота: ресемпл до 24 кГц делает сервер. Достаточно 5–20 секунд чистой речи одного диктора.',
    'voices.id': 'Идентификатор',
    'voices.file': 'WAV-файл',
    'voices.warm': 'Прогреть',
    'voices.warming': 'Прогреваю',
    'voices.prefix': 'Префикс',
    'voices.size': 'Размер',
    'voices.created': 'Создан',
    'voices.download': 'Скачать',
    'voices.delete': 'Удалить',
    'voices.empty': 'Пока ни одного голоса',
    'voices.deleteTitle': 'Удалить голос?',
    'voices.deleteBody': 'Файл состояния будет стёрт без возможности восстановления.',
    'voices.cancel': 'Отмена',
    'voices.badId': 'Идентификатор: латиница, цифры, дефис, подчёркивание.',
    'voices.noFile': 'Выберите WAV с образцом голоса.',
    'voices.warmed': 'Голос прогрет',

    'monitor.title': 'Мониторинг',
    'monitor.description': 'Воркеры по NUMA-нодам, перцентили задержек и счётчики.',
    'monitor.workers': 'Воркеры',
    'monitor.node': 'NUMA-нода',
    'monitor.status': 'Состояние',
    'monitor.served': 'Обслужено',
    'monitor.busy': 'занят',
    'monitor.idle': 'свободен',
    'monitor.latency': 'Задержки, мс',
    'monitor.request': 'Запрос целиком',
    'monitor.counters': 'Счётчики',
    'monitor.requests': 'Запросов',
    'monitor.errors': 'Ошибок',
    'monitor.queued': 'В очереди',
    'monitor.synthesized': 'Синтезировано',
    'monitor.rtfTotal': 'RTF совокупный',

    'notif.title': 'Уведомления',
    'notif.empty': 'Здесь появятся прошлые уведомления',
    'notif.clear': 'Очистить',

    'theme.light': 'Светлая',
    'theme.dark': 'Тёмная',
    'theme.system': 'Системная',
    'error.generic': 'Ошибка',
  },
  en: {
    'nav.synth': 'Synthesis',
    'nav.models': 'Models',

    'models.title': 'Models',
    'models.description': 'What is installed, what can be downloaded, and what is speaking now.',
    'models.installed': 'Installed',
    'models.available': 'Available',
    'models.available.hint':
      'Only checkpoints released as ONNX can be downloaded here. A Pocket TTS model cannot: it has to be traced from its weights, which needs the Python toolchain in install.sh.',
    'models.allInstalled': 'Everything in the catalogue is installed.',
    'models.jobs': 'Downloads',
    'models.active': 'active',
    'models.activate': 'Switch to',
    'models.install': 'Download',
    'models.installing': 'Download started',
    'models.switched': 'Model switched',
    'models.removed': 'Model removed',
    'models.cancel': 'Cancel',
    'models.cancelled': 'Download cancelled',
    'models.col.model': 'Model',
    'models.col.arch': 'Architecture',
    'models.col.rate': 'Rate',
    'models.col.voices': 'Voices',
    'models.col.size': 'Size',
    'models.col.langs': 'Languages',
    'models.job.running': 'running',
    'models.job.done': 'done',
    'models.job.failed': 'failed',
    'models.job.cancelled': 'cancelled',
    'models.disabled.title': 'Registry disabled',
    'models.disabled.body':
      'This server was started without --models, so it cannot install or switch models while running.',

    'synth.guidance': 'Guidance',
    'synth.durationScale': 'Pace',
    'synth.archNote.pocket':
      'Autoregressive: audio starts on the first frame, and latency does not grow with the text.',
    'synth.archNote.s3':
      'The sampler solves the whole utterance, so time to first sound grows with the length of the text.',

    'voices.cloneUnsupported':
      'This model ships fixed style vectors and cannot learn a voice from a recording.',
    'nav.voices': 'Voices',
    'nav.monitor': 'Monitoring',
    'app.subtitle': 'Russian speech synthesis on CPU',

    'conn.online': 'connected',
    'conn.offline': 'offline',
    'conn.auth': 'key required',
    'conn.checking': 'checking',

    'key.label': 'API key',
    'key.placeholder': 'Bearer key, if the service requires one',
    'key.hint': 'Kept in this browser only and sent as an Authorization header.',
    'key.save': 'Save',

    'synth.title': 'Synthesis',
    'synth.description': 'Time to first frame, and synthesis time over audio duration.',
    'synth.text': 'Text',
    'synth.textHint': 'Stress: a plus before the vowel, or U+0301 after it',
    'synth.voice': 'Voice',
    'synth.format': 'Format',
    'synth.temperature': 'Temperature',
    'synth.eos': 'EOS threshold',
    'synth.seed': 'Seed',
    'synth.firstChunk': 'First chunk, frames',
    'synth.run': 'Synthesize',
    'synth.running': 'Synthesizing',
    'synth.request': 'Ready-made request',
    'synth.copy': 'Copy',
    'synth.copied': 'Copied',
    'synth.pcmNote': 'Raw PCM received. The browser will not play it; latency was measured.',

    'metric.ttfb': 'TTFB',
    'metric.ttfbSub': 'to first frame',
    'metric.total': 'Total',
    'metric.totalSub': 'wall clock',
    'metric.audio': 'Audio',
    'metric.rtf': 'RTF',
    'metric.frames': 'frames',
    'metric.realtime': 'real time',

    'voices.title': 'Voices',
    'voices.description': 'Warmed FlowLM states, swapped in per request instead of a prefill.',
    'voices.warmTitle': 'Warm a new voice',
    'voices.warmHint':
      'The reference runs through Mimi and a FlowLM prefill; the result is a KV state in safetensors. Mono or stereo, any rate — the server resamples to 24 kHz. Five to twenty seconds of one clean speaker is enough.',
    'voices.id': 'Identifier',
    'voices.file': 'WAV file',
    'voices.warm': 'Warm',
    'voices.warming': 'Warming',
    'voices.prefix': 'Prefix',
    'voices.size': 'Size',
    'voices.created': 'Created',
    'voices.download': 'Download',
    'voices.delete': 'Delete',
    'voices.empty': 'No voices yet',
    'voices.deleteTitle': 'Delete this voice?',
    'voices.deleteBody': 'The state file will be removed and cannot be recovered.',
    'voices.cancel': 'Cancel',
    'voices.badId': 'Identifier: letters, digits, hyphen, underscore.',
    'voices.noFile': 'Choose a WAV with the voice sample.',
    'voices.warmed': 'Voice warmed',

    'monitor.title': 'Monitoring',
    'monitor.description': 'Workers per NUMA node, latency percentiles and counters.',
    'monitor.workers': 'Workers',
    'monitor.node': 'NUMA node',
    'monitor.status': 'Status',
    'monitor.served': 'Served',
    'monitor.busy': 'busy',
    'monitor.idle': 'idle',
    'monitor.latency': 'Latency, ms',
    'monitor.request': 'Whole request',
    'monitor.counters': 'Counters',
    'monitor.requests': 'Requests',
    'monitor.errors': 'Errors',
    'monitor.queued': 'Queued',
    'monitor.synthesized': 'Synthesized',
    'monitor.rtfTotal': 'Aggregate RTF',

    'notif.title': 'Notifications',
    'notif.empty': 'Past notifications will show up here',
    'notif.clear': 'Clear',

    'theme.light': 'Light',
    'theme.dark': 'Dark',
    'theme.system': 'System',
    'error.generic': 'Error',
  },
} as const

export type MessageKey = keyof (typeof dictionary)['ru']

interface I18nValue {
  locale: Locale
  setLocale: (l: Locale) => void
  t: (key: MessageKey) => string
}

const I18nContext = createContext<I18nValue | null>(null)

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(
    () => (safeLocal('nanotts.lang', 'ru') as Locale) ?? 'ru',
  )

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next)
    storeLocal('nanotts.lang', next)
    document.documentElement.lang = next
  }, [])

  const value = useMemo<I18nValue>(
    () => ({ locale, setLocale, t: (key) => dictionary[locale][key] }),
    [locale, setLocale],
  )
  return <I18nContext value={value}>{children}</I18nContext>
}

export function useI18n(): I18nValue {
  const value = use(I18nContext)
  if (!value) throw new Error('useI18n must be used inside I18nProvider')
  return value
}
