from typing import Callable

import uuid
import json
import urllib.request
import urllib.parse


server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt: dict) -> dict:
    print("prompting...")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def interrupt():
    print("interrupt")
    urllib.request.Request("http://{}/interrupt".format(server_address), method="POST")

def get_image(filename, subfolder, folder_type):
    print("get image...")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(
        ws,
        prompt: dict,
        preview_callback: Callable | None = None,
):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    need_stop = False
    out = None
    current_step = 0

    while True:
        prev_out = out
        out = ws.recv()
        if isinstance(out, str):
            if "execut" not in out:
                continue

            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break # Execution is done
        else:
            if preview_callback and "progress" in prev_out and prompt_id in prev_out:
                current_step += 1
                need_stop = preview_callback(out[8:], current_step)
                if need_stop:
                    break

    if not need_stop:
        history = get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images

# prompt = json.load(open("./api_template.json"))
