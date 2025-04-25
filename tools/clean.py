import io
import os
import sys
import re
import shutil
import mimetypes
import time
import asyncio
import subprocess
import secrets
from io import BytesIO
from PIL import Image
from aiohttp import web
from collections import deque
from imagehash import phash
from manga_translator import MangaTranslator

### --- concurrency patch (1) ---
# 默认并发协程数，可用环境变量 MT_CONCURRENCY 覆盖
CONCURRENCY = int(os.getenv("MT_CONCURRENCY", 4))
### ------------------------------

SERVER_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = os.path.dirname(os.path.dirname(SERVER_DIR_PATH))

# TODO: Get capabilities through api
VALID_LANGUAGES = {
    'CHS': 'Chinese (Simplified)',
    'CHT': 'Chinese (Traditional)',
    'CSY': 'Czech',
    'NLD': 'Dutch',
    'ENG': 'English',
    'FRA': 'French',
    'DEU': 'German',
    'HUN': 'Hungarian',
    'ITA': 'Italian',
    'JPN': 'Japanese',
    'KOR': 'Korean',
    'PLK': 'Polish',
    'PTB': 'Portuguese (Brazil)',
    'ROM': 'Romanian',
    'RUS': 'Russian',
    'ESP': 'Spanish',
    'TRK': 'Turkish',
    'UKR': 'Ukrainian',
    'VIN': 'Vietnamese',
    'ARA': 'Arabic',
}
# Whitelists
VALID_DETECTORS = set(['default', 'ctd'])
VALID_DIRECTIONS = set(['auto', 'h', 'v'])
VALID_TRANSLATORS = [
    'youdao',
    'baidu',
    'google',
    'deepl',
    'deepseek',
    'papago',
    'caiyun',
    'gpt3.5',
    'gpt4',
    'nllb',
    'nllb_big',
    'sugoi',
    'jparacrawl',
    'jparacrawl_big',
    'm2m100',
    'm2m100_big',
    'qwen2',
    'qwen2_big',
    'sakura',
    'none',
    'original',
]

### --- concurrency patch (1-b) ---
MAX_ONGOING_TASKS = CONCURRENCY     # 原来为 1
### -------------------------------
MAX_IMAGE_SIZE_PX = 8000**2

# Time to wait for web client to send a request to /task-state request
# before that web clients task gets removed from the queue
WEB_CLIENT_TIMEOUT = -1

# Time before finished tasks get removed from memory
FINISHED_TASK_REMOVE_TIMEOUT = 1800

# Auto deletes old task folders upon reaching this disk space limit
DISK_SPACE_LIMIT = 5e7 # 50mb

# TODO: Turn into dict with translator client id as key for support of multiple translator clients
ONGOING_TASKS = []
FINISHED_TASKS = []
NONCE = ''
QUEUE = deque()
TASK_DATA = {}
TASK_STATES = {}
DEFAULT_TRANSLATION_PARAMS = {}
AVAILABLE_TRANSLATORS = []
FORMAT = ''

app = web.Application(client_max_size = 1024 * 1024 * 50)
routes = web.RouteTableDef()


def constant_compare(a, b):
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    if not isinstance(a, bytes) or not isinstance(b, bytes):
        return False
    if len(a) != len(b):
        return False

    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0

@routes.get("/")
async def index_async(request):
    global AVAILABLE_TRANSLATORS
    with open(os.path.join(SERVER_DIR_PATH, 'ui.html'), 'r', encoding='utf8') as fp:
        content = fp.read()
        if AVAILABLE_TRANSLATORS:
            content = re.sub(r'(?<=translator: )(.*)(?=,)', repr(AVAILABLE_TRANSLATORS[0]), content)
            content = re.sub(r'(?<=validTranslators: )(\[.*\])(?=,)', repr(AVAILABLE_TRANSLATORS), content)
        return web.Response(text=content, content_type='text/html')

@routes.get("/manual")
async def index_async(request):
    with open(os.path.join(SERVER_DIR_PATH, 'manual.html'), 'r', encoding='utf8') as fp:
        return web.Response(text=fp.read(), content_type='text/html')

@routes.get("/result/{taskid}")
async def result_async(request):
    global FORMAT
    filepath = os.path.join('result', request.match_info.get('taskid'), f'final.{FORMAT}')
    if not os.path.exists(filepath):
        return web.Response(status=404, text='Not Found')
    stream = BytesIO()
    with open(filepath, 'rb') as f:
        stream.write(f.read())
    mime = mimetypes.guess_type(filepath)[0] or 'application/octet-stream'
    return web.Response(body=stream.getvalue(), content_type=mime)

@routes.get("/result-type")
async def file_type_async(request):
    global FORMAT
    return web.Response(text=f'{FORMAT}')

@routes.get("/queue-size")
async def queue_size_async(request):
    return web.json_response({'size' : len(QUEUE)})

async def handle_post(request):
    data = await request.post()
    detection_size = None
    selected_translator = 'youdao'
    target_language = 'CHS'
    detector = 'default'
    direction = 'auto'
    if 'target_lang' in data:
        target_language = data['target_lang'].upper()
        # TODO: move dicts to their own files to reduce load time
        if target_language not in VALID_LANGUAGES:
            target_language = 'CHS'
    if 'detector' in data:
        detector = data['detector'].lower()
        if detector not in VALID_DETECTORS:
            detector = 'default'
    if 'direction' in data:
        direction = data['direction'].lower()
        if direction not in VALID_DIRECTIONS:
            direction = 'auto'
    if 'translator' in data:
        selected_translator = data['translator'].lower()
        if selected_translator not in AVAILABLE_TRANSLATORS:
            selected_translator = AVAILABLE_TRANSLATORS[0]
    if 'size' in data:
        size_text = data['size'].upper()
        if size_text == 'S':
            detection_size = 1024
        elif size_text == 'M':
            detection_size = 1536
        elif size_text == 'L':
            detection_size = 2048
        elif size_text == 'X':
            detection_size = 2560
    if 'file' in data:
        file_field = data['file']
        content = file_field.file.read()
    elif 'url' in data:
        from aiohttp import ClientSession
        async with ClientSession() as session:
            async with session.get(data['url']) as resp:
                if resp.status == 200:
                    content = await resp.read()
                else:
                    return web.json_response({'status': 'error'})
    else:
        return web.json_response({'status': 'error'})
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        img = Image.open(io.BytesIO(content))
        if img.width * img.height > MAX_IMAGE_SIZE_PX:
            return web.json_response({'status': 'error-too-large'})
    except Exception:
        return web.json_response({'status': 'error-img-corrupt'})
    return img, detection_size, selected_translator, target_language, detector, direction

@routes.post("/run")
async def run_async(request):
    global FORMAT
    x = await handle_post(request)
    if isinstance(x, tuple):
        img, size, selected_translator, target_language, detector, direction = x
    else:
        return x
    timestamp = str(int(time.time() * 1000))  # 使用时间戳
    task_id = f'{size}-{selected_translator}-{target_language}-{detector}-{direction}-{timestamp}-{phash(img, hash_size = 16)}'
    print(f'New `run` task {task_id}')
    if os.path.exists(f'result/{task_id}/final.{FORMAT}'):
        # Add a console output prompt to avoid the console from appearing to be stuck without execution when the translated image is hit consecutively.
        print(f'Using cached result for {task_id}')
        return web.json_response({'task_id' : task_id, 'status': 'successful'})
    # elif os.path.exists(f'result/{task_id}'):
    #     # either image is being processed or error occurred
    #     if task_id not in TASK_STATES:
    #         # error occurred
    #         return web.json_response({'state': 'error'})
    else:
        os.makedirs(f'result/{task_id}/', exist_ok=True)
        img.save(f'result/{task_id}/input.png')
        QUEUE.append(task_id)
        now = time.time()
        TASK_DATA[task_id] = {
            'detection_size': size,
            'translator': selected_translator,
            'target_lang': target_language,
            'detector': detector,
            'direction': direction,
            'created_at': now,
            'requested_at': now,
        }
        TASK_STATES[task_id] = {
            'info': 'pending',
            'finished': False,
        }
    while True:
        await asyncio.sleep(0.1)
        if task_id not in TASK_STATES:
            break
        state = TASK_STATES[task_id]
        if state['finished']:
            break
    return web.json_response({'task_id': task_id, 'status': 'successful' if state['finished'] else state['info']})


@routes.post("/connect-internal")
async def index_async(request):
    global NONCE, VALID_TRANSLATORS, AVAILABLE_TRANSLATORS
    # Can be extended to allow support for multiple translators
    rqjson = await request.json()
    if constant_compare(rqjson.get('nonce'), NONCE):
        capabilities = rqjson.get('capabilities')
        if capabilities:
            translators = capabilities.get('translators')
            AVAILABLE_TRANSLATORS.clear()
            for key in VALID_TRANSLATORS:
                if key in translators:
                    AVAILABLE_TRANSLATORS.append(key)
    return web.json_response({})

@routes.get("/task-internal")
async def get_task_async(request):
    """
    Called by the translator to get a translation task.
    """
    global NONCE, ONGOING_TASKS, DEFAULT_TRANSLATION_PARAMS
    if constant_compare(request.rel_url.query.get('nonce'), NONCE):
        if len(QUEUE) > 0 and len(ONGOING_TASKS) < MAX_ONGOING_TASKS:
            task_id = QUEUE.popleft()
            if task_id in TASK_DATA:
                data = TASK_DATA[task_id]
                for p, default_value in DEFAULT_TRANSLATION_PARAMS.items():
                    current_value = data.get(p)
                    data[p] = current_value if current_value is not None else default_value
                if not TASK_DATA[task_id].get('manual', False):
                    ONGOING_TASKS.append(task_id)
                return web.json_response({'task_id': task_id, 'data': data})
            else:
                return web.json_response({})
        else:
            return web.json_response({})
    return web.json_response({})

async def manual_trans_task(task_id, texts, translations):
    if task_id not in TASK_DATA:
        TASK_DATA[task_id] = {}
    if texts and translations:
        TASK_DATA[task_id]['trans_request'] = [{'s': txt, 't': trans} for txt, trans in zip(texts, translations)]
    else:
        TASK_DATA[task_id]['trans_result'] = []
        print('Manual translation complete')

@routes.post("/cancel-manual-request")
async def cancel_manual_translation(request):
    rqjson = (await request.json())
    if 'task_id' in rqjson:
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            TASK_DATA[task_id]['cancel'] = ' '
            while True:
                await asyncio.sleep(0.1)
                if TASK_STATES[task_id]['info'].startswith('error'):
                    ret = web.json_response({'task_id': task_id, 'status': 'error'})
                    break
                if TASK_STATES[task_id]['finished']:
                    ret = web.json_response({'task_id': task_id, 'status': 'cancelled'})
                    break
            del TASK_STATES[task_id]
            del TASK_DATA[task_id]
            return ret
    return web.json_response({})

@routes.post("/post-manual-result")
async def post_translation_result(request):
    rqjson = (await request.json())
    if 'trans_result' in rqjson and 'task_id' in rqjson:
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            trans_result = [r['t'] for r in rqjson['trans_result']]
            TASK_DATA[task_id]['trans_result'] = trans_result
            while True:
                await asyncio.sleep(0.1)
                if TASK_STATES[task_id]['info'].startswith('error'):
                    ret = web.json_response({'task_id': task_id, 'status': 'error'})
                    break
                if TASK_STATES[task_id]['finished']:
                    ret = web.json_response({'task_id': task_id, 'status': 'successful'})
                    break
            # remove old tasks
            del TASK_STATES[task_id]
            del TASK_DATA[task_id]
            return ret
    return web.json_response({})

@routes.post("/request-manual-internal")
async def request_translation_internal(request):
    global NONCE
    rqjson = await request.json()
    if constant_compare(rqjson.get('nonce'), NONCE):
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            if TASK_DATA[task_id].get('manual', False):
                # manual translation
                asyncio.gather(manual_trans_task(task_id, rqjson['texts'], rqjson['translations']))
    return web.json_response({})

@routes.post("/get-manual-result-internal")
async def get_translation_internal(request):
    global NONCE
    rqjson = (await request.json())
    if constant_compare(rqjson.get('nonce'), NONCE):
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            if 'trans_result' in TASK_DATA[task_id]:
                return web.json_response({'result': TASK_DATA[task_id]['trans_result']})
            elif 'cancel' in TASK_DATA[task_id]:
                return web.json_response({'cancel':''})
    return web.json_response({})

@routes.get("/task-state")
async def get_task_state_async(request):
    """
    Web API for getting the state of an on-going translation task from the website.

    Is periodically called from ui.html. Once it returns a finished state,
    the web client will try to fetch the corresponding image through /result/<task_id>
    """
    task_id = request.query.get('taskid')
    if task_id and task_id in TASK_STATES and task_id in TASK_DATA:
        state = TASK_STATES[task_id]
        data = TASK_DATA[task_id]
        res_dict = {
            'state': state['info'],
            'finished': state['finished'],
        }
        data['requested_at'] = time.time()
        try:
            res_dict['waiting'] = QUEUE.index(task_id) + 1
        except Exception:
            res_dict['waiting'] = 0
        res = web.json_response(res_dict)

        return res
    return web.json_response({'state': 'error'})

@routes.post("/task-update-internal")
async def post_task_update_async(request):
    """
    Lets the translator update the task state it is working on.
    """
    global NONCE, ONGOING_TASKS, FINISHED_TASKS
    rqjson = (await request.json())
    if constant_compare(rqjson.get('nonce'), NONCE):
        task_id = rqjson['task_id']
        if task_id in TASK_STATES and task_id in TASK_DATA:
            TASK_STATES[task_id] = {
                'info': rqjson['state'],
                'finished': rqjson['finished'],
            }
            if rqjson['finished'] and not TASK_DATA[task_id].get('manual', False):
                try:
                    i = ONGOING_TASKS.index(task_id)
                    FINISHED_TASKS.append(ONGOING_TASKS.pop(i))
                except ValueError:
                    pass
            print(f'Task state {task_id} to {TASK_STATES[task_id]}')
    return web.json_response({})

@routes.post("/submit")
async def submit_async(request):
    """Adds new task to the queue. Called by web client in ui.html when submitting an image."""
    global FORMAT
    x = await handle_post(request)
    if isinstance(x, tuple):
        img, size, selected_translator, target_language, detector, direction = x
    else:
        return x
    timestamp = str(int(time.time() * 1000))  # 使用时间戳
    task_id = f'{size}-{selected_translator}-{target_language}-{detector}-{direction}-{timestamp}-{phash(img, hash_size = 16)}'
    now = time.time()
    print(f'New `submit` task {task_id}')
    if os.path.exists(f'result/{task_id}/final.{FORMAT}'):
        TASK_STATES[task_id] = {
            'info': 'saved',
            'finished': True,
        }
        TASK_DATA[task_id] = {
            'detection_size': size,
            'translator': selected_translator,
            'target_lang': target_language,
            'detector': detector,
            'direction': direction,
            'created_at': now,
            'requested_at': now,
        }
    elif task_id not in TASK_DATA or task_id not in TASK_STATES:
        os.makedirs(f'result/{task_id}/', exist_ok=True)
        img.save(f'result/{task_id}/input.png')
        QUEUE.append(task_id)
        TASK_STATES[task_id] = {
            'info': 'pending',
            'finished': False,
        }
        TASK_DATA[task_id] = {
            'detection_size': size,
            'translator': selected_translator,
            'target_lang': target_language,
            'detector': detector,
            'direction': direction,
            'created_at': now,
            'requested_at': now,
        }
    return web.json_response({'task_id': task_id, 'status': 'successful'})

@routes.post("/manual-translate")
async def manual_translate_async(request):
    x = await handle_post(request)
    if isinstance(x, tuple):
        img, size, selected_translator, target_language, detector, direction = x
    else:
        return x
    task_id = secrets.token_hex(16)
    print(f'New `manual-translate` task {task_id}')
    os.makedirs(f'result/{task_id}/', exist_ok=True)
    img.save(f'result/{task_id}/input.png')
    now = time.time()
    QUEUE.append(task_id)
    # TODO: Add form fields to manual translate website
    TASK_DATA[task_id] = {
        # 'detection_size': size,
        'manual': True,
        # 'detector': detector,
        # 'direction': direction,
        'created_at': now,
        'requested_at': now,
    }
    print(TASK_DATA[task_id])
    TASK_STATES[task_id] = {
        'info': 'pending',
        'finished': False,
    }
    while True:
        await asyncio.sleep(1)
        if 'trans_request' in TASK_DATA[task_id]:
            return web.json_response({'task_id' : task_id, 'status': 'pending', 'trans_result': TASK_DATA[task_id]['trans_request']})
        if TASK_STATES[task_id]['info'].startswith('error'):
            break
        if TASK_STATES[task_id]['finished']:
            # no texts detected
            return web.json_response({'task_id' : task_id, 'status': 'successful'})
    return web.json_response({'task_id' : task_id, 'status': 'error'})

app.add_routes(routes)


def generate_nonce():
    return secrets.token_hex(16)

def start_translator_client_proc(host: str, port: int, nonce: str, params: dict):
    os.environ['MT_WEB_NONCE'] = nonce
    cmds = [
        sys.executable,
        '-m', 'manga_translator',
        '--mode', 'web_client',
        '--host', host,
        '--port', str(port),
    ]
    if params.get('use_gpu', False):
        cmds.append('--use-gpu')
    if params.get('use_gpu_limited', False):
        cmds.append('--use-gpu-limited')
    if params.get('ignore_errors', False):
        cmds.append('--ignore-errors')
    if params.get('verbose', False):
        cmds.append('--verbose')

    proc = subprocess.Popen(cmds, cwd=BASE_PATH)
    return proc

async def start_async_app(host: str, port: int, nonce: str, translation_params: dict = None):
    global NONCE, DEFAULT_TRANSLATION_PARAMS, FORMAT
    # Secret to secure communication between webserver and translator clients
    NONCE = nonce
    DEFAULT_TRANSLATION_PARAMS = translation_params or {}
    FORMAT = DEFAULT_TRANSLATION_PARAMS.get('format') or 'jpg'
    DEFAULT_TRANSLATION_PARAMS['format'] = FORMAT

    # Schedule web server to run
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f'Serving up app on http://{host}:{port}')

    return runner, site

async def dispatch(host: str, port: int, nonce: str = None, translation_params: dict = None):
    """
    启动 aiohttp Web 服务器，并在同一进程内用协程并发翻译任务。
    """
    global ONGOING_TASKS, FINISHED_TASKS

    if nonce is None:
        nonce = os.getenv('MT_WEB_NONCE', generate_nonce())

    # 开启 Web 服务
    runner, site = await start_async_app(host, port, nonce, translation_params)

    # --- ① 仅加载一次模型 ---
    translator = MangaTranslator(translation_params or {})
    sem = asyncio.Semaphore(CONCURRENCY)

    async def progress_hook(state, finished):
        if tid := getattr(asyncio.current_task(), "tid", None):
            TASK_STATES[tid]['info'] = state
            TASK_STATES[tid]['finished'] = finished

    translator.add_progress_hook(progress_hook)

    async def worker():
        while True:
            # 等待任务
            task_id = await asyncio.get_running_loop().run_in_executor(None, QUEUE.popleft)
            asyncio.current_task().tid = task_id
            async with sem:
                try:
                    inp = os.path.join('result', task_id, 'input.png')
                    out = os.path.join('result', task_id, f'final.{FORMAT}')
                    params = TASK_DATA[task_id]
                    TASK_STATES[task_id]['info'] = 'detection'
                    await translator.translate_path(inp, out, params)
                except Exception as e:
                    TASK_STATES[task_id]['info'] = 'error'
                    TASK_STATES[task_id]['finished'] = True
                    print(f"[{task_id}] translate failed:", e)
                else:
                    TASK_STATES[task_id]['info'] = 'saved'
                    TASK_STATES[task_id]['finished'] = True

    # 启动 N 个协程 worker
    workers = [asyncio.create_task(worker()) for _ in range(CONCURRENCY)]

    # ---------- 与原版一致的回收 / 清理循环 ----------
    try:
        os.makedirs('result/', exist_ok=True)
        for f in os.listdir('result/'):
            if os.path.isdir(f'result/{f}') and re.search(r'^\w+-\d+-\w+-\w+-\w+-\w+$', f):
                FINISHED_TASKS.append(f)
        FINISHED_TASKS.sort(key=lambda tid: os.path.getmtime(f'result/{tid}'))

        while True:
            await asyncio.sleep(1)

            # 过期任务 & 磁盘清理（原逻辑）
            now = time.time()
            del_ids = [tid for tid, s in TASK_STATES.items()
                       if s['finished'] and now - TASK_DATA[tid]['created_at'] > FINISHED_TASK_REMOVE_TIMEOUT]
            for tid in del_ids:
                TASK_STATES.pop(tid, None)
                TASK_DATA.pop(tid, None)

            if DISK_SPACE_LIMIT >= 0 and FINISHED_TASKS and shutil.disk_usage('result/')[2] < DISK_SPACE_LIMIT:
                tid = FINISHED_TASKS.pop(0)
                try:
                    shutil.rmtree(f'result/{tid}')
                except FileNotFoundError:
                    pass
    finally:
        for w in workers:
            w.cancel()
        await runner.cleanup()
### ----------------------  end patch  ----------------------

if __name__ == '__main__':
    from ..args import parser

    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        runner, site = loop.run_until_complete(dispatch(args.host, args.port, translation_params=vars(args)))
    except KeyboardInterrupt:
        pass
