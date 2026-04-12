import numpy as np
import os
import re
import datetime
import arxiv
import openai, tenacity
import base64, requests
import argparse
import configparser
import fitz, io, os
from PIL import Image
import gradio
import markdown
import json
import tiktoken
import concurrent.futures
from optimizeOpenAI import chatPaper
import ipywidgets as widgets
from IPython.display import display

def parse_text(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            items = line.split('`')
            if items[-1]:
                lines[i] = f'<pre><code class="{items[-1]}">'
            else:
                lines[i] = f'</code></pre>'
        else:
            if i > 0:
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                lines[i] = '<br/>' + line.replace(" ", "&nbsp;")
    return "".join(lines)


# def get_response(system, context, myKey, raw = False):
#     openai.api_key = myKey
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[system, *context],
#     )
#     openai.api_key = ""
#     if raw:
#         return response
#     else:
#         message = response["choices"][0]["message"]["content"]
#         message_with_stats = f'{message}'
#         return message, parse_text(message_with_stats)

valid_api_keys = []


def api_key_check(api_key):
    try:
        chat = chatPaper([api_key])
        if chat.check_api_available():
            return api_key
        else:
            return None
    except:
        return None


def valid_apikey(api_keys):
    api_keys = api_keys.replace(' ', '')
    api_key_list = api_keys.split(',')
    print(api_key_list)
    global valid_api_keys
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_results = {
            executor.submit(api_key_check, api_key): api_key
            for api_key in api_key_list
        }
        for future in concurrent.futures.as_completed(future_results):
            result = future.result()
            if result:
                valid_api_keys.append(result)
    if len(valid_api_keys) > 0:
        return "有效的api-key一共有{}个，分别是：{}, 现在可以提交你的paper".format(
            len(valid_api_keys), valid_api_keys)
    return "无效的api-key"


class Paper:

    def __init__(self, path, title='', url='', abs='', authers=[], sl=[]):
        # 初始化函数，根据pdf路径初始化Paper对象
        self.url = url  # 文章链接
        self.path = path  # pdf路径
        self.sl = sl
        self.section_names = []  # 段落标题
        self.section_texts = {}  # 段落内容
        self.abs = abs
        self.title_page = 0
        if title == '':
            self.pdf = fitz.open(self.path)  # pdf文档
            self.title = self.get_title()
            self.parse_pdf()
        else:
            self.title = title
        self.authers = authers
        self.roman_num = [
            "I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"
        ]
        self.digit_num = [str(d + 1) for d in range(10)]
        self.first_image = ''

    def parse_pdf(self):
        self.pdf = fitz.open(self.path)  # pdf文档
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ' '.join(self.text_list)
        self.section_page_dict = self._get_all_page_index()  # 段落与页码的对应字典
        print("section_page_dict", self.section_page_dict)
        self.section_text_dict = self._get_all_page()  # 段落与内容的对应字典
        self.section_text_dict.update({"title": self.title})
        self.section_text_dict.update({"paper_info": self.get_paper_info()})
        self.pdf.close()

    def get_paper_info(self):
        first_page_text = self.pdf[self.title_page].get_text()
        if "Abstract" in self.section_text_dict.keys():
            abstract_text = self.section_text_dict['Abstract']
        else:
            abstract_text = self.abs
        introduction_text = self.section_text_dict['Introduction']
        first_page_text = first_page_text.replace(abstract_text, "").replace(
            introduction_text, "")
        return first_page_text

    def get_image_path(self, image_path=''):
        """
        将PDF中的第一张图保存到image.png里面，存到本地目录，返回文件名称，供gitee读取
        :param filename: 图片所在路径，"C:\\Users\\Administrator\\Desktop\\nwd.pdf"
        :param image_path: 图片提取后的保存路径
        :return:
        """
        # open file
        max_size = 0
        image_list = []
        with fitz.Document(self.path) as my_pdf_file:
            # 遍历所有页面
            for page_number in range(1, len(my_pdf_file) + 1):
                # 查看独立页面
                page = my_pdf_file[page_number - 1]
                # 查看当前页所有图片
                images = page.get_images()
                # 遍历当前页面所有图片
                for image_number, image in enumerate(page.get_images(),
                                                     start=1):
                    # 访问图片xref
                    xref_value = image[0]
                    # 提取图片信息
                    base_image = my_pdf_file.extract_image(xref_value)
                    # 访问图片
                    image_bytes = base_image["image"]
                    # 获取图片扩展名
                    ext = base_image["ext"]
                    # 加载图片
                    image = Image.open(io.BytesIO(image_bytes))
                    image_size = image.size[0] * image.size[1]
                    if image_size > max_size:
                        max_size = image_size
                    image_list.append(image)
        for image in image_list:
            image_size = image.size[0] * image.size[1]
            if image_size == max_size:
                image_name = f"image.{ext}"
                im_path = os.path.join(image_path, image_name)
                print("im_path:", im_path)

                max_pix = 480
                origin_min_pix = min(image.size[0], image.size[1])

                if image.size[0] > image.size[1]:
                    min_pix = int(image.size[1] * (max_pix / image.size[0]))
                    newsize = (max_pix, min_pix)
                else:
                    min_pix = int(image.size[0] * (max_pix / image.size[1]))
                    newsize = (min_pix, max_pix)
                image = image.resize(newsize)

                image.save(open(im_path, "wb"))
                return im_path, ext
        return None, None

    # 定义一个函数，根据字体的大小，识别每个章节名称，并返回一个列表
    def get_chapter_names(self, ):
        # # 打开一个pdf文件
        doc = fitz.open(self.path)  # pdf文档
        text_list = [page.get_text() for page in doc]
        all_text = ''
        for text in text_list:
            all_text += text
        # # 创建一个空列表，用于存储章节名称
        chapter_names = []
        for line in all_text.split('\n'):
            line_list = line.split(' ')
            if '.' in line:
                point_split_list = line.split('.')
                space_split_list = line.split(' ')
                if 1 < len(space_split_list) < 5:
                    if 1 < len(point_split_list) < 5 and (
                            point_split_list[0] in self.roman_num
                            or point_split_list[0] in self.digit_num):
                        print("line:", line)
                        chapter_names.append(line)

        return chapter_names

    def get_title(self):
        doc = self.pdf  # 打开pdf文件
        max_font_size = 0  # 初始化最大字体大小为0
        max_string = ""  # 初始化最大字体大小对应的字符串为空
        max_font_sizes = [0]
        for page_index, page in enumerate(doc):  # 遍历每一页
            text = page.get_text("dict")  # 获取页面上的文本信息
            blocks = text["blocks"]  # 获取文本块列表
            for block in blocks:  # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']):  # 如果是文字类型
                    if len(block["lines"][0]["spans"]):
                        font_size = block["lines"][0]["spans"][0][
                            "size"]  # 获取第一行第一段文字的字体大小
                        max_font_sizes.append(font_size)
                        if font_size > max_font_size:  # 如果字体大小大于当前最大值
                            max_font_size = font_size  # 更新最大值
                            max_string = block["lines"][0]["spans"][0][
                                "text"]  # 更新最大值对应的字符串
        max_font_sizes.sort()
        print("max_font_sizes", max_font_sizes[-10:])
        cur_title = ''
        for page_index, page in enumerate(doc):  # 遍历每一页
            text = page.get_text("dict")  # 获取页面上的文本信息
            blocks = text["blocks"]  # 获取文本块列表
            for block in blocks:  # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']):  # 如果是文字类型
                    if len(block["lines"][0]["spans"]):
                        cur_string = block["lines"][0]["spans"][0][
                            "text"]  # 更新最大值对应的字符串
                        font_flags = block["lines"][0]["spans"][0][
                            "flags"]  # 获取第一行第一段文字的字体特征
                        font_size = block["lines"][0]["spans"][0][
                            "size"]  # 获取第一行第一段文字的字体大小
                        # print(font_size)
                        if abs(font_size - max_font_sizes[-1]) < 0.3 or abs(
                                font_size - max_font_sizes[-2]) < 0.3:
                            # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)
                            if len(cur_string
                                   ) > 4 and "arXiv" not in cur_string:
                                # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)
                                if cur_title == '':
                                    cur_title += cur_string
                                else:
                                    cur_title += ' ' + cur_string
                            self.title_page = page_index

        title = cur_title.replace('\n', ' ')
        return title

    def _get_all_page_index(self):
        # 定义需要寻找的章节名称列表
        section_list = self.sl
        # 初始化一个字典来存储找到的章节和它们在文档中出现的页码
        section_page_dict = {}
        # 遍历每一页文档
        for page_index, page in enumerate(self.pdf):
            # 获取当前页面的文本内容
            cur_text = page.get_text()
            # 遍历需要寻找的章节名称列表
            for section_name in section_list:
                # 将章节名称转换成大写形式
                section_name_upper = section_name.upper()
                # 如果当前页面包含"Abstract"这个关键词
                if "Abstract" == section_name and section_name in cur_text:
                    # 将"Abstract"和它所在的页码加入字典中
                    section_page_dict[section_name] = page_index
                # 如果当前页面包含章节名称，则将章节名称和它所在的页码加入字典中
                else:
                    if section_name + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
                    elif section_name_upper + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
        # 返回所有找到的章节名称及它们在文档中出现的页码
        return section_page_dict

    def _get_all_page(self):
        """
        获取PDF文件中每个页面的文本信息，并将文本信息按照章节组织成字典返回。
        Returns:
            section_dict (dict): 每个章节的文本信息字典，key为章节名，value为章节文本。
        """
        text = ''
        text_list = []
        section_dict = {}

        # 再处理其他章节：
        text_list = [page.get_text() for page in self.pdf]
        for sec_index, sec_name in enumerate(self.section_page_dict):
            print(sec_index, sec_name, self.section_page_dict[sec_name])
            if sec_index <= 0 and self.abs:
                continue
            else:
                # 直接考虑后面的内容：
                start_page = self.section_page_dict[sec_name]
                if sec_index < len(list(self.section_page_dict.keys())) - 1:
                    end_page = self.section_page_dict[list(
                        self.section_page_dict.keys())[sec_index + 1]]
                else:
                    end_page = len(text_list)
                print("start_page, end_page:", start_page, end_page)
                cur_sec_text = ''
                if end_page - start_page == 0:
                    if sec_index < len(list(
                            self.section_page_dict.keys())) - 1:
                        next_sec = list(
                            self.section_page_dict.keys())[sec_index + 1]
                        if text_list[start_page].find(sec_name) == -1:
                            start_i = text_list[start_page].find(
                                sec_name.upper())
                        else:
                            start_i = text_list[start_page].find(sec_name)
                        if text_list[start_page].find(next_sec) == -1:
                            end_i = text_list[start_page].find(
                                next_sec.upper())
                        else:
                            end_i = text_list[start_page].find(next_sec)
                        cur_sec_text += text_list[start_page][start_i:end_i]
                else:
                    for page_i in range(start_page, end_page):
                        #                         print("page_i:", page_i)
                        if page_i == start_page:
                            if text_list[start_page].find(sec_name) == -1:
                                start_i = text_list[start_page].find(
                                    sec_name.upper())
                            else:
                                start_i = text_list[start_page].find(sec_name)
                            cur_sec_text += text_list[page_i][start_i:]
                        elif page_i < end_page:
                            cur_sec_text += text_list[page_i]
                        elif page_i == end_page:
                            if sec_index < len(
                                    list(self.section_page_dict.keys())) - 1:
                                next_sec = list(
                                    self.section_page_dict.keys())[sec_index +
                                                                   1]
                                if text_list[start_page].find(next_sec) == -1:
                                    end_i = text_list[start_page].find(
                                        next_sec.upper())
                                else:
                                    end_i = text_list[start_page].find(
                                        next_sec)
                                cur_sec_text += text_list[page_i][:end_i]
                section_dict[sec_name] = cur_sec_text.replace('-\n',
                                                              '').replace(
                                                                  '\n', ' ')
        return section_dict


# 定义Reader类
class Reader:
    # 初始化方法，设置属性
    def __init__(self,
                 key_word='',
                 query='',
                 filter_keys='',
                 root_path='./',
                 gitee_key='',
                 sort=arxiv.SortCriterion.SubmittedDate,
                 user_name='defualt',
                 language='cn',
                 api_keys: list = [],
                 model_name="gpt-3.5-turbo",
                 p=1.0,
                 temperature=1.0):
        self.api_keys = api_keys
        self.chatPaper = chatPaper(api_keys=self.api_keys,
                                   apiTimeInterval=10,
                                   temperature=temperature,
                                   top_p=p,
                                   model_name=model_name)  #openAI api封装
        self.user_name = user_name  # 读者姓名
        self.key_word = key_word  # 读者感兴趣的关键词
        self.query = query  # 读者输入的搜索查询
        self.sort = sort  # 读者选择的排序方式
        self.language = language  # 读者选择的语言
        self.filter_keys = filter_keys  # 用于在摘要中筛选的关键词
        self.root_path = root_path
        self.file_format = 'md'  # or 'txt'，如果为图片，则必须为'md'
        self.save_image = False
        if self.save_image:
            self.gitee_key = self.config.get('Gitee', 'api')
        else:
            self.gitee_key = ''
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")

    def get_arxiv(self, max_results=30):
        search = arxiv.Search(
            query=self.query,
            max_results=max_results,
            sort_by=self.sort,
            sort_order=arxiv.SortOrder.Descending,
        )
        return search

    def filter_arxiv(self, max_results=30):
        search = self.get_arxiv(max_results=max_results)
        print("all search:")
        for index, result in enumerate(search.results()):
            print(index, result.title, result.updated)

        filter_results = []
        filter_keys = self.filter_keys

        print("filter_keys:", self.filter_keys)
        # 确保每个关键词都能在摘要中找到，才算是目标论文
        for index, result in enumerate(search.results()):
            abs_text = result.summary.replace('-\n', '-').replace('\n', ' ')
            meet_num = 0
            for f_key in filter_keys.split(" "):
                if f_key.lower() in abs_text.lower():
                    meet_num += 1
            if meet_num == len(filter_keys.split(" ")):
                filter_results.append(result)
                # break
        print("filter_results:", len(filter_results))
        print("filter_papers:")
        for index, result in enumerate(filter_results):
            print(index, result.title, result.updated)
        return filter_results

    def validateTitle(self, title):
        # 将论文的乱七八糟的路径格式修正
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title)  # 替换为下划线
        return new_title

    def download_pdf(self, filter_results):
        # 先创建文件夹
        date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
        key_word = str(self.key_word.replace(':', ' '))
        path = self.root_path + 'pdf_files/' + self.query.replace(
            'au: ', '').replace('title: ', '').replace('ti: ', '').replace(
                ':', ' ')[:25] + '-' + date_str
        try:
            os.makedirs(path)
        except:
            pass
        print("All_paper:", len(filter_results))
        # 开始下载：
        paper_list = []
        for r_index, result in enumerate(filter_results):
            try:
                title_str = self.validateTitle(result.title)
                pdf_name = title_str + '.pdf'
                # result.download_pdf(path, filename=pdf_name)
                self.try_download_pdf(result, path, pdf_name)
                paper_path = os.path.join(path, pdf_name)
                print("paper_path:", paper_path)
                paper = Paper(
                    path=paper_path,
                    url=result.entry_id,
                    title=result.title,
                    abs=result.summary.replace('-\n', '-').replace('\n', ' '),
                    authers=[str(aut) for aut in result.authors],
                )
                # 下载完毕，开始解析：
                paper.parse_pdf()
                paper_list.append(paper)
            except Exception as e:
                print("download_error:", e)
                pass
        return paper_list

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4,
                                                   max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def try_download_pdf(self, result, path, pdf_name):
        result.download_pdf(path, filename=pdf_name)

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4,
                                                   max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def upload_gitee(self, image_path, image_name='', ext='png'):
        """
        上传到码云
        :return:
        """
        with open(image_path, 'rb') as f:
            base64_data = base64.b64encode(f.read())
            base64_content = base64_data.decode()

        date_str = str(datetime.datetime.now())[:19].replace(':', '-').replace(
            ' ', '-') + '.' + ext
        path = image_name + '-' + date_str

        payload = {
            "access_token": self.gitee_key,
            "owner": self.config.get('Gitee', 'owner'),
            "repo": self.config.get('Gitee', 'repo'),
            "path": self.config.get('Gitee', 'path'),
            "content": base64_content,
            "message": "upload image"
        }
        # 这里需要修改成你的gitee的账户和仓库名，以及文件夹的名字：
        url = f'https://gitee.com/api/v5/repos/' + self.config.get(
            'Gitee', 'owner') + '/' + self.config.get(
                'Gitee', 'repo') + '/contents/' + self.config.get(
                    'Gitee', 'path') + '/' + path
        rep = requests.post(url, json=payload).json()
        print("rep:", rep)
        if 'content' in rep.keys():
            image_url = rep['content']['download_url']
        else:
            image_url = r"https://gitee.com/api/v5/repos/" + self.config.get(
                'Gitee', 'owner') + '/' + self.config.get(
                    'Gitee', 'repo') + '/contents/' + self.config.get(
                        'Gitee', 'path') + '/' + path

        return image_url


    def summary_with_chat(self, paper_list):
        htmls = []
        utoken = 0
        ctoken = 0
        ttoken = 0
        for paper_index, paper in enumerate(paper_list):
            # 第一步先用title，abs，和introduction进行总结。
            text = ''
            text += 'Title:' + paper.title
            text += 'Url:' + paper.url
            text += 'Abstrat:' + paper.abs
            text += 'Paper_info:' + paper.section_text_dict['paper_info']
            # intro
            text += list(paper.section_text_dict.values())[0]
            #max_token = 2500 * 4
            #text = text[:max_token]
            chat_summary_text, utoken1, ctoken1, ttoken1 = self.chat_summary(
                text=text)
            htmls.append(chat_summary_text)

            # TODO 往md文档中插入论文里的像素最大的一张图片，这个方案可以弄的更加智能一些：
            method_key = ''
            for parse_key in paper.section_text_dict.keys():
                if 'method' in parse_key.lower(
                ) or 'approach' in parse_key.lower():
                    method_key = parse_key
                    break

            if method_key != '':
                text = ''
                method_text = ''
                summary_text = ''
                summary_text += "<summary>" + chat_summary_text
                # methods
                method_text += paper.section_text_dict[method_key]
                text = summary_text + "\n<Methods>:\n" + method_text
                chat_method_text, utoken2, ctoken2, ttoken2 = self.chat_method(
                    text=text)
            else:
                chat_method_text = ''
            htmls.append(chat_method_text)
            htmls.append("\n")

            # 第三步总结全文，并打分：
            conclusion_key = ''
            for parse_key in paper.section_text_dict.keys():
                if 'conclu' in parse_key.lower():
                    conclusion_key = parse_key
                    break

            text = ''
            conclusion_text = ''
            summary_text = ''
            summary_text += "<summary>" + chat_summary_text + "\n <Method summary>:\n" + chat_method_text
            if conclusion_key != '':
                # conclusion
                conclusion_text += paper.section_text_dict[conclusion_key]
                text = summary_text + "\n <Conclusion>:\n" + conclusion_text
            else:
                text = summary_text
            chat_conclusion_text, utoken3, ctoken3, ttoken3 = self.chat_conclusion(
                text=text)
            htmls.append(chat_conclusion_text)
            htmls.append("\n")
            # token统计
            utoken = utoken + utoken1 + utoken2 + utoken3
            ctoken = ctoken + ctoken1 + ctoken2 + ctoken3
            ttoken = ttoken + ttoken1 + ttoken2 + ttoken3
            cost = (ttoken / 1000) * 0.002
            pos_count = {
                "usage_token_used": str(utoken),
                "completion_token_used": str(ctoken),
                "total_token_used": str(ttoken),
                "cost": str(cost),
            }
            md_text = "\n".join(htmls)
            return markdown.markdown(md_text), pos_count

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4,
                                                   max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_conclusion(self, text):
        conclusion_prompt_token = 650
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(
            len(text) * (self.max_token_num - conclusion_prompt_token) /
            text_token)
        clip_text = text[:clip_text_index]
        self.chatPaper.reset(
            convo_id="chatConclusion",
            system_prompt="You are a reviewer specializing in [" +
            self.key_word + "], with strong background in EEG emotion decoding, EEG foundation models, general EEG decoding, time-series modeling, contrastive learning, transfer learning, knowledge distillation, domain adaptation, and domain generalization. You need to critically review this article and judge whether its ideas can transfer to EEG emotion decoding.")
        self.chatPaper.add_to_conversation(
            convo_id="chatConclusion",
            role="assistant",
            message=
            "This is the <summary> and <conclusion> part of an English paper. The <summary> has already been written. Please focus on the real contribution, limitations, generalization value, and whether this work is useful for EEG emotion decoding or adjacent EEG/time-series research: "
            + clip_text)  # 背景知识，可以参考OpenReview的审稿流程
        content = """                 
                 8. Make the following summary. Be sure to use Chinese answers (proper nouns need to be marked in English).
                    - (1): What is the significance of this work for its original task or field?
                    - (2): Summarize the strengths and weaknesses of this article in four dimensions: innovation, empirical performance, robustness/generalization, and workload/reproducibility.
                    - (3): If I want to borrow this work for EEG emotion decoding, what are the main risks or limitations? Consider subject shift, session shift, label scarcity, noise, dataset scale, and computational cost.
                    - (4): Give a final recommendation for my research use: directly follow, selectively borrow, or only keep as background reading. State the most reusable idea clearly.
                 Follow the format of the output later: 
                 8. Conclusion: \n\n
                    - (1):xxx;\n                     
                    - (2):Innovation: xxx; Performance: xxx; Robustness/Generalization: xxx; Workload/Reproducibility: xxx;\n
                    - (3):xxx;\n
                    - (4):xxx;\n
                 
                 Be sure to use Chinese answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.                 
                 """
        result = self.chatPaper.ask(
            prompt=content,
            role="user",
            convo_id="chatConclusion",
        )
        print(result)
        return result[0], result[1], result[2], result[3]

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4,
                                                   max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_method(self, text):
        method_prompt_token = 650
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(
            len(text) * (self.max_token_num - method_prompt_token) /
            text_token)
        clip_text = text[:clip_text_index]
        self.chatPaper.reset(
            convo_id="chatMethod",
            system_prompt="You are a researcher specializing in [" +
            self.key_word +
            "], with strong background in EEG emotion decoding, EEG foundation models, general EEG decoding, time-series modeling, contrastive learning, transfer learning, knowledge distillation, domain adaptation, and domain generalization. You are good at turning paper methods into compact and reusable research notes."
        )  # chatgpt 角色
        self.chatPaper.add_to_conversation(
            convo_id="chatMethod",
            role="assistant",
            message=str(
                "This is the <summary> and <Method> part of an English paper. The <summary> has already been written. Please focus on the method pipeline, the training objective, the generalization design, and what can transfer to EEG emotion decoding: "
                + clip_text))
        content = """                 
                 7. Describe the methodological idea of this article in detail. Be sure to use Chinese answers (proper nouns need to be marked in English).
                    - (1): What is the input form and preprocessing pipeline? Mention signal type, segmentation/windowing, filtering, augmentation, and whether the representation is raw EEG, handcrafted features, time-frequency maps, graphs, or tokenized sequences when available.
                    - (2): What is the backbone or encoder architecture? Explain the key temporal, spatial, channel, frequency, graph, or multimodal modeling modules.
                    - (3): What is the training objective and optimization strategy? Mention classification/regression losses and whether contrastive learning, self-supervised learning, transfer learning, knowledge distillation, domain adaptation, or domain generalization is used.
                    - (4): What is the complete method pipeline in ordered steps?
                    - (5): What are the data requirements, computational characteristics, or deployment constraints if the paper mentions them?
                    - (6): Which parts are most likely to transfer to EEG emotion decoding or cross-subject EEG generalization?
                 Follow the format of the output that follows: 
                 7. Methods: \n\n
                    - (1):xxx;\n 
                    - (2):xxx;\n 
                    - (3):xxx;\n  
                    - (4):xxx;\n
                    - (5):xxx;\n
                    - (6):xxx;\n\n
                 
                 Be sure to use Chinese answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.                 
                 """
        result = self.chatPaper.ask(
            prompt=content,
            role="user",
            convo_id="chatMethod",
        )
        print(result)
        return result[0], result[1], result[2], result[3]

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4,
                                                   max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_summary(self, text):
        summary_prompt_token = 1000
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(
            len(text) * (self.max_token_num - summary_prompt_token) /
            text_token)
        clip_text = text[:clip_text_index]
        self.chatPaper.reset(
            convo_id="chatSummary",
            system_prompt="You are a researcher specializing in [" +
            self.key_word +
            "], with strong background in EEG emotion decoding, EEG foundation models, general EEG decoding, time-series modeling, contrastive learning, transfer learning, knowledge distillation, domain adaptation, and domain generalization. You summarize papers concisely, structurally, and critically, and you explicitly judge whether a paper is directly useful, partially transferable, or mainly inspirational for EEG emotion decoding.")
        self.chatPaper.add_to_conversation(
            convo_id="chatSummary",
            role="assistant",
            message=str(
                "This is the title, author, link, abstract and introduction of an English paper. Please read it as a researcher deciding whether the paper is useful for EEG emotion decoding or adjacent EEG/time-series research: "
                + clip_text))
        content = """                 
                 1. Mark the title of the paper (with Chinese translation)
                 2. List all authors' names (use English)
                 3. Mark the first author's affiliation (output Chinese translation only)                 
                 4. Mark the keywords of this article (use English)
                 5. Link to the paper and Github/code link (if available, fill in Github:None if not)
                 6. Summarize according to the following six points. Be sure to use Chinese answers (proper nouns need to be marked in English)
                    - (1): What problem does this paper study, and how is it related to EEG emotion decoding, general EEG decoding, or adjacent time-series research?
                    - (2): What task setting, dataset, evaluation protocol, and subject/session split are used? If the paper is not EEG-specific, state the real task setting clearly.
                    - (3): What are the representative previous methods and their limitations? Why is the new method needed?
                    - (4): What is the core methodology of this paper? Pay special attention to time-series modeling, contrastive learning, transfer learning, knowledge distillation, domain adaptation/generalization, or foundation-model style pretraining if they appear.
                    - (5): What performance is achieved, on which benchmarks or metrics, and are the gains practically meaningful?
                    - (6): How useful is this paper for my research on EEG emotion decoding? State whether it is directly usable, partially transferable, or mainly inspirational, and explain why.
                 Follow the format of the output that follows:                  
                 1. Title: xxx\n\n
                 2. Authors: xxx\n\n
                 3. Affiliation: xxx\n\n                 
                 4. Keywords: xxx\n\n   
                 5. Urls: xxx or xxx , xxx \n\n      
                 6. Summary: \n\n
                    - (1):xxx;\n 
                    - (2):xxx;\n 
                    - (3):xxx;\n  
                    - (4):xxx;\n
                    - (5):xxx;\n
                    - (6):xxx.\n\n     
                 
                 Be sure to use Chinese answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed.                 
                 """
        result = self.chatPaper.ask(
            prompt=content,
            role="user",
            convo_id="chatSummary",
        )
        print(result)
        return result[0], result[1], result[2], result[3]

    def export_to_markdown(self, text, file_name, mode='w'):
        # 使用markdown模块的convert方法，将文本转换为html格式
        # html = markdown.markdown(text)
        # 打开一个文件，以写入模式
        with open(file_name, mode, encoding="utf-8") as f:
            # 将html格式的内容写入文件
            f.write(text)

    # 定义一个方法，打印出读者信息
    def show_info(self):
        print(f"Key word: {self.key_word}")
        print(f"Query: {self.query}")
        print(f"Sort: {self.sort}")


def upload_pdf(api_keys, text, model_name, p, temperature, file):
    # 检查两个输入都不为空
    api_key_list = None
    if api_keys:
        api_key_list = api_keys.split(',')
    elif not api_keys and valid_api_keys != []:
        api_key_list = valid_api_keys
    if not text or not file or not api_key_list:
        return "两个输入都不能为空，请输入字符并上传 PDF 文件！"

    # 判断PDF文件
    #if file and file.name.split(".")[-1].lower() != "pdf":
    #    return '请勿上传非 PDF 文件！'
    else:
        section_list = text.split(',')
        paper_list = [Paper(path=file, sl=section_list)]
        # 创建一个Reader对象
        print(api_key_list)
        reader = Reader(api_keys=api_key_list,
                        model_name=model_name,
                        p=p,
                        temperature=temperature)
        sum_info, cost = reader.summary_with_chat(
            paper_list=paper_list)  # type: ignore
        return cost, sum_info


api_title = "api-key可用验证"
api_description = '''<div align='left'>
<img src='https://visitor-badge.laobi.icu/badge?page_id=https://huggingface.co/spaces/wangrongsheng/ChatPaper'>
<img align='right' src='https://i.328888.xyz/2023/03/12/vH9dU.png' width="150">
Use ChatGPT to summary the papers.Star our Github [🌟ChatPaper](https://github.com/kaixindelele/ChatPaper) .
💗如果您觉得我们的项目对您有帮助，还请您给我们一些鼓励！💗
🔴请注意：千万不要用于严肃的学术场景，只能用于论文阅读前的初筛！
</div>
'''

api_input = [
    gradio.inputs.Textbox(label="请输入你的API-key(必填, 多个API-key请用英文逗号隔开)",
                          default="",
                          type='password')
]
api_gui = gradio.Interface(fn=valid_apikey,
                           inputs=api_input,
                           outputs="text",
                           title=api_title,
                           description=api_description)

# 标题
title = "ChatPaper"
# 描述
description = '''<div align='left'>
<img src='https://visitor-badge.laobi.icu/badge?page_id=https://huggingface.co/spaces/wangrongsheng/ChatPaper'>
<img align='right' src='https://i.328888.xyz/2023/03/12/vH9dU.png' width="150">
Use ChatGPT to summary the papers.Star our Github [🌟ChatPaper](https://github.com/kaixindelele/ChatPaper) .
💗如果您觉得我们的项目对您有帮助，还请您给我们一些鼓励！💗
🔴请注意：千万不要用于严肃的学术场景，只能用于论文阅读前的初筛！
</div>
'''
# 创建Gradio界面
ip = [
    gradio.inputs.Textbox(label="请输入你的API-key(必填, 多个API-key请用英文逗号隔开),不需要空格",
                          default="",
                          type='password'),
    gradio.inputs.Textbox(
        label="请输入论文大标题索引(用英文逗号隔开,必填)",
        default=
        "'Abstract,Introduction,Related Work,Background,Preliminary,Problem Formulation,Methods,Methodology,Method,Approach,Approaches,Materials and Methods,Experiment Settings,Experiment,Experimental Results,Evaluation,Experiments,Results,Findings,Data Analysis,Discussion,Results and Discussion,Conclusion,References'"
    ),
    gradio.inputs.Radio(choices=["gpt-3.5-turbo", "gpt-3.5-turbo-0301"],
                        default="gpt-3.5-turbo",
                        label="Select model"),
    gradio.inputs.Slider(minimum=-0,
                         maximum=1.0,
                         default=1.0,
                         step=0.05,
                         label="Top-p (nucleus sampling)"),
    gradio.inputs.Slider(minimum=-0,
                         maximum=5.0,
                         default=0.5,
                         step=0.5,
                         label="Temperature"),
    gradio.inputs.File(label="请上传论文PDF(必填)")
]


chatpaper_gui = gradio.Interface(fn=upload_pdf,
                                 inputs=ip,
                                 outputs=["json", "html"],
                                 title=title,
                                 description=description)

# Start server
gui = gradio.TabbedInterface(interface_list=[api_gui, chatpaper_gui],
                             tab_names=["API-key", "ChatPaper"])
gui.launch(quiet=True, show_api=False)
