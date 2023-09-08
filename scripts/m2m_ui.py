import os
import shutil
import sys

import gradio as gr
import platform
import modules.scripts as scripts
from modules import script_callbacks, shared, ui_postprocessing, call_queue, sd_samplers
from modules.call_queue import wrap_gradio_gpu_call
from modules.sd_samplers import samplers_for_img2img
from modules.shared import opts, cmd_opts
from modules.ui import paste_symbol, clear_prompt_symbol, extra_networks_symbol, apply_style_symbol, save_style_symbol, \
    create_refresh_button, create_sampler_and_steps_selection, ordered_ui_categories, switch_values_symbol, \
    create_override_settings_dropdown
from modules.ui_common import folder_symbol, plaintext_to_html
from modules.ui_components import ToolButton, FormRow, FormGroup
import modules.generation_parameters_copypaste as parameters_copypaste
import subprocess as sp
from scripts import mov2mov
from scripts.m2m_config import mov2mov_outpath_samples, mov2mov_output_dir
from scripts.m2m_modnet import modnet_models

html_id = "mov2mov"
# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001F5D1'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅

def create_seed_inputs(target_interface):
    with FormRow(elem_id=f"{target_interface}_seed_row", variant="compact"):
        if cmd_opts.use_textbox_seed:
            seed = gr.Textbox(label='Seed', value="", elem_id=f"{target_interface}_seed")
        else:
            seed = gr.Number(label='Seed', value=-1, elem_id=f"{target_interface}_seed", precision=0)
        random_seed = ToolButton(random_symbol, elem_id=f"{target_interface}_random_seed", label='Random seed')
        reuse_seed = ToolButton(reuse_symbol, elem_id=f"{target_interface}_reuse_seed", label='Reuse seed')
        seed_checkbox = gr.Checkbox(label='Extra', elem_id=f"{target_interface}_subseed_show", value=False)
    # Components to show/hide based on the 'Extra' checkbox
    seed_extras = []
    with FormRow(visible=False, elem_id=f"{target_interface}_subseed_row") as seed_extra_row_1:
        seed_extras.append(seed_extra_row_1)
        subseed = gr.Number(label='Variation seed', value=-1, elem_id=f"{target_interface}_subseed", precision=0)
        random_subseed = ToolButton(random_symbol, elem_id=f"{target_interface}_random_subseed")
        reuse_subseed = ToolButton(reuse_symbol, elem_id=f"{target_interface}_reuse_subseed")
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=f"{target_interface}_subseed_strength")
    with FormRow(visible=False) as seed_extra_row_2:
        seed_extras.append(seed_extra_row_2)
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from width", value=0, elem_id=f"{target_interface}_seed_resize_from_w")
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from height", value=0, elem_id=f"{target_interface}_seed_resize_from_h")
    random_seed.click(fn=None, _js="function(){setRandomSeed('" + target_interface + "_seed')}", show_progress=False, inputs=[], outputs=[])
    random_subseed.click(fn=None, _js="function(){setRandomSeed('" + target_interface + "_subseed')}", show_progress=False, inputs=[], outputs=[])
    def change_visibility(show):
        return {comp: gr_show(show) for comp in seed_extras}
    seed_checkbox.change(change_visibility, show_progress=False, inputs=[seed_checkbox], outputs=seed_extras)
    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox

def create_toprow():
    with gr.Row(elem_id=f"{html_id}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{html_id}_prompt_container", scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{html_id}_prompt", show_label=False, lines=3,
                                            placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)")

            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{html_id}_neg_prompt",
                                                     show_label=False, lines=2,
                                                     placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)")

        with gr.Column(scale=1, elem_id=f"{html_id}_actions_column"):
            with gr.Row(elem_id=f"{html_id}_generate_box"):
                interrupt = gr.Button('Interrupt', elem_id=f"{html_id}_interrupt")
                skip = gr.Button('Skip', elem_id=f"{html_id}_skip")
                submit = gr.Button('Generate', elem_id=f"{html_id}_generate", variant='primary')

                # add copy from txt2img img2img

                skip.click(
                    fn=lambda: shared.state.skip(),
                    inputs=[],
                    outputs=[],
                )

                interrupt.click(
                    fn=lambda: shared.state.interrupt(),
                    inputs=[],
                    outputs=[],
                )

            # with gr.Row(elem_id=f'{html_id}_copy'):
            #     copy_from_txt2img = gr.Button('copy from txt2img', elem_id=f"{html_id}_copy_from_txt2img",
            #                                   variant='secondary')
            #
            #     copy_from_img2img = gr.Button('copy from img2img', elem_id=f"{html_id}_copy_from_img2img",
            #                                   variant='secondary')
            #
            #     copy_from_txt2img.click(None, [], [], _js="() => {return copy_from('txt2img')}")
            #
            #     copy_from_img2img.click(None, [], [], _js="() => {return copy_from('img2img')}")

    return prompt, negative_prompt, submit


def save_video(video):
    path = 'logs/movies'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    index = len([path for path in os.listdir(path) if path.endswith('.mp4')]) + 1
    video_path = os.path.join(path, str(index).zfill(5) + '.mp4')
    shutil.copyfile(video, video_path)
    filename = os.path.relpath(video_path, path)
    return gr.File.update(value=video_path, visible=True), plaintext_to_html(f"Saved: {filename}")


def create_output_panel(tabname, outdir):
    from modules import shared
    import modules.generation_parameters_copypaste as parameters_copypaste

    def open_folder(f):
        if not os.path.exists(f):
            print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
            return
        elif not os.path.isdir(f):
            print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
            return

        if not shared.cmd_opts.hide_ui_dir_config:
            path = os.path.normpath(f)
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                sp.Popen(["open", path])
            elif "microsoft-standard-WSL2" in platform.uname().release:
                sp.Popen(["wsl-open", path])
            else:
                sp.Popen(["xdg-open", path])

    with gr.Column(variant='panel', elem_id=f"{tabname}_results"):
        with gr.Group(elem_id=f"{tabname}_gallery_container"):
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"{tabname}_gallery").style(grid=4)
            result_video = gr.Video(label='Output Video', show_label=False, elem_id=f'{tabname}_video', interactive=False)

        generation_info = None
        with gr.Column():
            with gr.Row(elem_id=f"image_buttons_{tabname}"):
                open_folder_button = gr.Button(folder_symbol,
                                               elem_id="hidden_element" if shared.cmd_opts.hide_ui_dir_config else f'open_folder_{tabname}')

                save = gr.Button('Save', elem_id=f'save_{tabname}')
                # save_zip = gr.Button('Zip', elem_id=f'save_zip_{tabname}')

            open_folder_button.click(
                fn=lambda: open_folder(shared.opts.outdir_samples or outdir),
                inputs=[],
                outputs=[],
            )

            with gr.Row():
                download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False,
                                         visible=False, elem_id=f'download_files_{tabname}')

            with gr.Group():
                html_info = gr.HTML(elem_id=f'html_info_{tabname}')
                html_log = gr.HTML(elem_id=f'html_log_{tabname}')

                generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')

                save.click(
                    fn=call_queue.wrap_gradio_call(save_video),
                    inputs=[
                        result_video
                    ],
                    outputs=[
                        download_files,
                        html_log,
                    ],
                    show_progress=False,
                )

        return result_gallery, result_video, generation_info, html_info, html_log


def create_modnet(html_id):
    with gr.Group():
        with gr.Accordion("ModNet", open=True):
            background_image = gr.Image(label='Background', type='numpy', elem_id='modnet_background_image').style()
            background_movie = gr.Video(label='Background', elem_id='modnet_background_movie').style()
            with gr.Row():
                gr.HTML(
                    value='''
                      <p>
                        Refer to the <a href="https://github.com/DavG25/sd-webui-mov2mov/wiki/ModNet" target="_blank" style="text-decoration: underline;">mov2mov wiki</a> for more information on how to use ModNet
                      </p>
                    ''')

            with gr.Row():
                enable = gr.Checkbox(label='Enable', value=False, elem_id=f"{html_id}_enable")  # 启用就是提取人物了

                modnet_model = gr.Dropdown(label='Model', choices=list(modnet_models), value='none',
                                           elem_id=f"{html_id}_modnet_model")

            with gr.Row():
                modnet_resize_mode = gr.Radio(label="Resize mode", elem_id=f"{html_id}_resize_mode",
                                              choices=["Just resize", "Crop and resize", "Resize and fill",
                                                       "Just resize (latent upscale)"], type="index",
                                              value="Just resize")

            with gr.Row():
                merge_background_mode = gr.Radio(label='Background mode', elem_id=f'{html_id}_merge_background_mode',
                                                 choices=['Clear', 'Origin', 'Green', 'Image', 'Video'],
                                                 type='index',
                                                 value='Clear')

                merge_background_mode.change(fn=None, inputs=[merge_background_mode], outputs=[],
                                             _js='switchModnetMode')

            with gr.Row():
                movie_frames = gr.Slider(minimum=10,
                                         maximum=60,
                                         step=1,
                                         label='Movie frames',
                                         elem_id='modnet_movie_frames',
                                         value=30)

    return [enable, background_image, background_movie, modnet_model, modnet_resize_mode, merge_background_mode,
            movie_frames]


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as mov2mov_tabs:
        dummy_component = gr.Label(visible=False)

        mov2mov_prompt, mov2mov_negative_prompt, submit = create_toprow()

        # create tabs
        with FormRow().style(equal_height=False):
            with gr.Column(variant='compact', elem_id=f"{html_id}_settings"):
                with gr.Tabs(elem_id=f"mode_{html_id}"):
                    with gr.TabItem('Input video', id='mov2mov', elem_id=f"{html_id}_mov2mov_tab") as tab_mov2mov:
                        init_mov = gr.Video(label="Video for mov2mov", elem_id=f"{html_id}_mov", show_label=False,
                                            source="upload")  # .style(height=480)
                        init_mov.change(None, None, None, _js="() => { refreshMov2movVideoHelper() }")

                # Video dimensions helper
                with FormRow(elem_id="mov2mov_input_video_info"):
                    gr.Textbox(label="Input video width", elem_id="mov2mov_input_video_info_width",
                               value="N/A", interactive=False)
                    gr.Textbox(label="Input video height", elem_id="mov2mov_input_video_info_height",
                               value="N/A", interactive=False)
                    

                with FormRow():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode",
                                           choices=["Just resize", "Crop and resize", "Resize and fill",
                                                    "Just resize (latent upscale)"], type="index", value="Just resize")

                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_name = create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(), "mov2mov")
                        # steps, sampler_name = create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(), "img2img")


                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id=f"{html_id}_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512,
                                                  elem_id=f"{html_id}_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512,
                                                   elem_id=f"{html_id}_height")

                            res_switch_btn = ToolButton(value=switch_values_symbol, elem_id=f"{html_id}_res_switch_btn")
                            with gr.Column(elem_id=f"{html_id}_column_batch"):
                                generate_mov_mode = gr.Radio(label="Video codec", elem_id="movie_codec",
                                                              choices=["MP4V", "H.264", "XVID", ], type="index",
                                                              value="H.264")

                                noise_multiplier = gr.Slider(minimum=0,
                                                              maximum=1.5,
                                                              step=0.01,
                                                              label='Noise multiplier',
                                                              elem_id=f'{html_id}_noise_multiplier',
                                                              value=0)

                                    # color_correction = gr.Checkbox(
                                    #     value=False,
                                    #     elem_id=f'{html_id}_color_correction',
                                    #     label='Color correction')

                    elif category == "cfg":
                        with FormGroup():
                            with FormRow():
                                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0,
                                                      elem_id=f"{html_id}_cfg_scale")
                                image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale',
                                                            value=1.5, elem_id=f"{html_id}_image_cfg_scale",
                                                            visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit")
                            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                           label='Denoising strength', value=0.75,
                                                           elem_id=f"{html_id}_denoising_strength")
                            movie_frames = gr.Slider(minimum=10,
                                                     maximum=60,
                                                     step=1,
                                                     label='Video frames',
                                                     elem_id=f'{html_id}_movie_frames',
                                                     value=30)

                    elif category == "seed":
                        max_frames = gr.Number(label='Max frames', value=-1, elem_id=f'{html_id}_max_frames')
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs(
                            'mov2mov')

                        seed.style(container=False)

                    elif category == "checkboxes":
                        with FormRow(elem_id=f"{html_id}_checkboxes", variant="compact"):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False,
                                                        visible=len(shared.face_restorers) > 1,
                                                        elem_id=f"{html_id}_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id=f"{html_id}_tiling")





                    elif category == "override_settings":
                        with FormRow(elem_id=f"{html_id}_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('mov2mov', row)

                    elif category == "scripts":
                        with FormGroup(elem_id=f"{html_id}_script_container"):
                            modnet_enable, modnet_background_image, modnet_background_movie, modnet_model, modnet_resize_mode, modnet_merge_background_mode, modnet_movie_frames = create_modnet(
                                'modnet')

                            custom_inputs = scripts.scripts_img2img.setup_ui()

            mov2mov_gallery, result_video, generation_info, html_info, html_log = create_output_panel("mov2mov",
                                                                                                      shared.opts.data.get(
                                                                                                          "mov2mov_output_dir",
                                                                                                          mov2mov_output_dir))

            mov2mov_args = dict(
                fn=wrap_gradio_gpu_call(mov2mov.mov2mov, extra_outputs=[None, '', '']),
                _js="submit_mov2mov",
                inputs=[
                           dummy_component,
                           # dummy_component, # mode
                           mov2mov_prompt,
                           mov2mov_negative_prompt,
                           init_mov,
                           steps,
                           sampler_name,
                           restore_faces,
                           tiling,
                           # extract_characters,
                           # merge_background,
                           # modnet_model,
                           modnet_enable, modnet_background_image, modnet_background_movie, modnet_model,
                           modnet_resize_mode, modnet_merge_background_mode, modnet_movie_frames,

                           generate_mov_mode,
                           noise_multiplier,
                           # color_correction,
                           cfg_scale,
                           image_cfg_scale,
                           denoising_strength,
                           movie_frames,
                           max_frames,
                           seed,
                           subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                           height,
                           width,
                           resize_mode,
                           override_settings,
                       ] + custom_inputs,
                outputs=[
                    mov2mov_gallery,
                    result_video,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )
            submit.click(**mov2mov_args)

    return [(mov2mov_tabs, "mov2mov", f"{html_id}_tabs")]


# 注册设置页的配置项
def on_ui_settings():
    section = ('mov2mov', "Mov2Mov")
    shared.opts.add_option("mov2mov_outpath_samples", shared.OptionInfo(
        mov2mov_outpath_samples, "Mov2Mov output path for image", section=section))  # 图片保存路径
    shared.opts.add_option("mov2mov_output_dir", shared.OptionInfo(
        mov2mov_output_dir, "Mov2Mov output path for video", section=section))  # 视频保存路径


script_callbacks.on_ui_settings(on_ui_settings)  # 注册进设置页
script_callbacks.on_ui_tabs(on_ui_tabs)
