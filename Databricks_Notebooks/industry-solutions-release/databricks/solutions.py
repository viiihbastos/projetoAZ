from databricks_api import DatabricksAPI
import shutil
import base64
from urllib.parse import quote, unquote
import json
import re
import os
from importlib import resources
import logging
import sys


head_section = re.compile("%md[\n\\s]*# (.*)\n?")
part_section = re.compile("%md[\n\\s]*## (.*)\n?")
notebook_regex = re.compile(
    "DATABRICKS_NOTEBOOK_MODEL = '((?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?)'")


def get_resource(module: str, name: str) -> str:
    """Load a textual resource file."""
    with resources.open_text(module, name, encoding="utf-8") as f:
        return f.read()


def create_readme_page(solution_name, content):
    content_markdown = base64.b64decode(content['content']).decode('UTF-8')
    content_markdown = f'%md {content_markdown}'
    json_object = json.loads(get_resource('databricks.resources', 'template_readme.json'))
    json_object['name'] = f'{solution_name} / README'
    json_object['commands'][0]['command'] = content_markdown
    return encode_json_to_notebook(json_object)


def persist_readme_page(solution_name, local_dir, content):

    html_text = get_resource('databricks.resources', 'template_readme.html')
    notebook_html = html_text.replace('[DATABRICKS_NOTEBOOK_MODEL]', content)

    with open(f'{local_dir}/{solution_name}.html', "w") as file_out:
        file_out.write(notebook_html)


def create_index_page(solution_name, index_href):

    json_object = json.loads(get_resource('databricks.resources', 'template_index.json'))
    json_object['name'] = solution_name

    cmd = '%md '
    for notebook_href in index_href:
        cmd = '{}\n\n{}'.format(cmd, notebook_href)

    json_object['commands'][0]['command'] = cmd
    json_str = json.dumps(json_object)
    json_str = quote(json_str).encode('utf-8')
    return base64.b64encode(json_str).decode('UTF-8')


def persist_index_page(solution_name, local_dir, index_notebook, landing_page):

    output_file = '{}/index.html'.format(local_dir)
    html_text = get_resource('databricks.resources', 'template_index.html')
    html_text = enrich_html(html_text)
    html_text = html_text.replace('[NOTEBOOK_HTML_LINK]', landing_page)
    html_text = html_text.replace('[NOTEBOOK_CONTENT]', index_notebook)
    html_text = html_text.replace('[SOLUTION_NAME]', solution_name)

    with open(output_file, "w") as f:
        f.write(html_text)


def get_section_name(section):
    cmd = section['command']
    if head_section.match(cmd):
        return head_section.match(cmd).group(1)
    elif part_section.match(cmd):
        return part_section.match(cmd).group(1)
    else:
        return 'Context'


def process_notebook_section(notebook_json, section_id, subsection_id, commands):
    notebook_json['commands'] = commands
    notebook_name = get_section_name(commands[0])
    notebook_encoded = encode_json_to_notebook(notebook_json)
    return Section(section_id, subsection_id, notebook_name, notebook_encoded)


def decode_notebook_to_json(notebook_content):
    org_notebook = base64.b64decode(notebook_content).decode('utf-8')
    return json.loads(unquote(org_notebook))


def encode_json_to_notebook(notebook_json):
    notebook_encoded = quote(json.dumps(notebook_json))
    return base64.b64encode(notebook_encoded.encode('utf-8')).decode('utf-8')


def process_notebook_content(notebook_id, notebook_content, notebook_name):
    notebook_json = decode_notebook_to_json(notebook_content)
    notebook_json['name'] = notebook_name

    commands = []
    notebooks = []
    section_id = notebook_id + 1
    subsection_id = 0

    for command in notebook_json['commands']:

        if not commands:
            commands = [command]
        else:
            command_value = command['command']
            # Start of a new top level section
            if head_section.match(command_value):
                new_notebook = process_notebook_section(notebook_json, section_id, subsection_id, commands)
                notebooks.append(new_notebook)
                commands = [command]
                subsection_id = 0
                section_id += 1
            # Start of a second level section
            elif part_section.match(command_value):
                new_notebook = process_notebook_section(notebook_json, section_id, subsection_id, commands)
                notebooks.append(new_notebook)
                commands = [command]
                subsection_id += 1
            else:
                # following of previous sections
                commands.append(command)

    new_notebook = process_notebook_section(notebook_json, section_id, subsection_id, commands)
    notebooks.append(new_notebook)
    return notebooks


def extract_content(html_content):
    matches = notebook_regex.findall(html_content)
    if matches:
        return matches[0]
    else:
        raise Exception("Could not extract notebook resources from HTML")


def transform_html(org_html, notebook_encoded):
    notebook_encoded = re.sub(
        notebook_regex,
        "DATABRICKS_NOTEBOOK_MODEL = '{}'".format(notebook_encoded),
        org_html
    )
    return notebook_encoded


def is_notebook(o):
    return o['object_type'] == "NOTEBOOK"


def valid_file(o):
    if is_notebook(o):
        if re.compile("^\\d+").match(os.path.basename(o['path'])):
            return True
    return False


def enrich_html(html_content):
    t = get_resource('databricks.resources', 'tag_header.html')
    html_content = html_content.replace('<head>', '<head>\n{}'.format(t))
    t = get_resource('databricks.resources', 'tag_body.html')
    html_content = html_content.replace('<body>', '<body>\n{}'.format(t))
    t = get_resource('databricks.resources', 'tag_body_end.html')
    html_content = html_content.replace('</body>', '{}\n</body>'.format(t))
    return html_content


def create_index_html_element(notebook_name, notebook_title):
    return '<a class=index href="{}">{}</a>'.format(notebook_name, notebook_title)


class Section:
    def __init__(self, section_id, subsection_id, notebook_name, notebook_encoded):
        self.section_id = section_id
        self.subsection_id = subsection_id
        self.notebook_name = notebook_name
        self.notebook_encoded = notebook_encoded
        self.logger = logging.getLogger('databricks')

    def html_name(self, solution_name):
        return f"{solution_name}_{self.section_id}-{self.subsection_id}.html"

    def get_number(self):
        if self.subsection_id == 0:
            return f"{self.section_id}"
        else:
            return f"{self.section_id}.{self.subsection_id}"


class Accelerator:

    def __init__(self, db_host, db_token, db_path, db_name, markdown=False):

        root = logging.getLogger()
        root.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)

        self.markdown = markdown
        self.db_path = db_path
        self.db_name = db_name
        self.db = DatabricksAPI(host=db_host, token=db_token)
        self.logger = logging.getLogger('databricks')

    def export_to_html(self, local_dir):

        self.logger.info("Exporting solution accelerator to HTML file(s)")
        db_objects = self.db.workspace.list(self.db_path)['objects']

        # Retrieve list of numbered notebooks. Those will be our core story telling assets
        db_notebooks = [db_object['path'] for db_object in db_objects if valid_file(db_object)]

        # Append list of numbered notebooks (story telling) with whatever additional util notebooks
        # Those would be added to the end of the index in alphabetical order
        for db_object in db_objects:
            db_path = db_object['path']
            if is_notebook(db_object) and db_path not in db_notebooks:
                db_notebooks.append(db_path)

        index_html = []
        landing_page = None
        section_id = 0

        if len(db_notebooks) == 0:
            raise Exception('Could not find any valid notebook in solution accelerator. Check naming convention')

        self.logger.info("Importing solution [README.md] file")
        readme_file = False
        for db_object in db_objects:
            if db_object['object_type'] == 'FILE' and db_object['path'].split('/')[-1] == 'README.md':
                readme_file = True
                readme_content = self.db.workspace.export_workspace(db_object['path'])
                readme_notebook = create_readme_page(self.db_name, readme_content)
                persist_readme_page(self.db_name, local_dir, readme_notebook)
                landing_page = f'{self.db_name}.html'
                index_html.append(create_index_html_element(landing_page, 'Context'))
                break
        if not readme_file:
            logging.warning("Could not find README file in solution")

        for i, file in enumerate(sorted(db_notebooks)):

            html_content = self.db.workspace.export_workspace(file, format='HTML')
            html_text = base64.b64decode(html_content['content']).decode('utf-8')

            self.logger.info("Processing notebook {} [{}]".format(i + 1, file.split('/')[-1]))

            if self.markdown:
                notebook_name = '{} / {}'.format(self.db_name, file.split('/')[-1])
                notebook = extract_content(html_text)
                children = process_notebook_content(section_id, notebook, notebook_name)
                for child in children:
                    if child.section_id > section_id:
                        section_id = child.section_id
                    child_name = child.html_name(self.db_name)
                    with open("{}/{}".format(local_dir, child_name), 'w') as f_out:
                        child_html = transform_html(html_text, child.notebook_encoded)
                        f_out.write(child_html)
                        child_title = "{} {}".format(child.get_number(), child.notebook_name)
                        index_html.append(create_index_html_element(child_name, child_title))
                        if not landing_page:
                            landing_page = child_name
            else:
                notebook_name = "{}_{}.html".format(self.db_name, i + 1)
                if not landing_page:
                    landing_page = notebook_name
                with open("{}/{}".format(local_dir, notebook_name), 'w') as f_out:
                    f_out.write(html_text)
                    index_html.append(create_index_html_element(notebook_name, file.split('/')[-1]))

        self.logger.info("Create Index page")
        index_notebook = create_index_page(self.db_name, index_html)
        persist_index_page(self.db_name, local_dir, index_notebook, landing_page)

        self.logger.info("Adding lead collector")
        with open("{}/{}".format(local_dir, 'lead-collector.js'), 'w') as f_out:
            f_out.write(get_resource('databricks.resources', 'lead-collector.js'))

    def release(self):
        output_dir = 'site'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        self.logger.info(f"Releasing solution [{self.db_name}]")
        self.export_to_html(output_dir)
