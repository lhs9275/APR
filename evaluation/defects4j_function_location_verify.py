from tree_sitter_languages import get_parser
import json
import os
import re
from pathlib import Path
from dataset_adapter import Defects4J

# Get Tree-sitter parser for Java
parser = get_parser("java")


def normalize_type(t):
    t = t.replace("final", "").replace("[]", "[]").strip()
    t = re.sub(r'\s+', '', t)  # remove all whitespace
    return t


def parse_method_path(method_path):
    pattern = r'(?:([\w:]+)::)?(\w+)\s*\((.*?)\)'
    match = re.match(pattern, method_path.strip())
    if not match:
        raise ValueError(f"Invalid method path: {method_path}")

    class_path_raw = match.group(1)
    method_name = match.group(2)
    param_string = match.group(3).strip()

    class_path = class_path_raw.split("::") if class_path_raw else []

    # Remove Java modifiers and extract just the type
    java_modifiers = {'final', 'static', 'public', 'private', 'protected', 'volatile', 'transient'}

    param_types = []
    if param_string.strip():
        for param in param_string.split(','):
            tokens = re.split(r'\s+', param.strip())
            tokens = [t for t in tokens if t not in java_modifiers]

            # Separate out the variable name (last token) from the type
            if len(tokens) < 2:
                continue  # skip malformed
            type_tokens = tokens[:-1]

            # Join and normalize things like "byte", "[", "]" -> "byte[]"
            joined = ' '.join(type_tokens).replace('[ ]', '[]')
            joined = re.sub(r'\s*\[\s*\]', '[]', joined)  # normalize all `[ ]` to `[]`
            joined = joined.replace(' ', '')  # remove remaining spaces in type
            param_types.append(joined)

    return class_path, method_name, param_types


def find_methods_with_name_and_params(tree, code_bytes, class_path, method_name_target, param_types_target):
    root_node = tree.root_node
    matches = []

    def visit(node, class_stack):
        if node.type == 'class_declaration':
            name_node = node.child_by_field_name('name')
            if name_node:
                class_stack = class_stack + [code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8").strip()]

        if node.type in ('method_declaration', 'constructor_declaration'):
            method_name_node = node.child_by_field_name('name')
            method_name = code_bytes[method_name_node.start_byte:method_name_node.end_byte].decode("utf-8").strip()
            if method_name != method_name_target:
                # print(f"method_name diff: {method_name, method_name_target}")
                return
            if class_stack[-len(class_path):] != class_path:
                # print(f"class diff： {class_stack, class_stack[-len(class_path):], class_path}")
                return

            parameters_node = node.child_by_field_name('parameters')
            method_param_types = []
            for child in parameters_node.children:
                if child.type == 'formal_parameter':
                    type_node = child.child_by_field_name('type')
                    if type_node:
                        param_type = code_bytes[type_node.start_byte:type_node.end_byte].decode("utf-8").strip()
                        method_param_types.append(normalize_type(param_type))

            if method_param_types != param_types_target:
                # print("parameter type diff")
                return

            matches.append(node)

        for child in node.children:
            visit(child, class_stack)

    visit(root_node, [])
    return matches


def get_method_boundaries(code, method_path):
    class_path, method_name, param_types = parse_method_path(method_path)
    print(f'{class_path, method_name, param_types}')
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    methods = find_methods_with_name_and_params(tree, code_bytes, class_path, method_name, param_types)
    if not methods:
        print(f"cannot find the method!")
        return None, None
    method_node = methods[0]

    # start after the annotation line
    method_start_node = None
    for child in method_node.children:
        if child.type not in ("modifier", "modifiers", "annotation"):
            method_start_node = child
            break

    start_line = method_start_node.start_point[0] + 1 if method_start_node else method_node.start_point[0] + 1
    end_line = method_node.end_point[0] + 1
    return start_line, end_line


def main():
    # Load JSON
    adapter = Defects4J()
    json_path = Path(__file__).resolve().parent.parent / "dataset" / "defects4j" / "defects4j_bugs_meta_data.json"
    with open(json_path, 'r') as f:
        bugs_data = json.load(f)

    updated = False
    base_checkout_dir = "/defects4j/framework/bin/temp/"

    error_cases = []
    not_same_cases = []
    for bug_id, bug_data in bugs_data.items():
        # debug
        # if str(bug_id) != '102':
        #     continue

        func_info = bug_data.get('function', {})
        function_after_start_line = func_info["function_after_start_line"]
        function_after_end_line = func_info["function_after_end_line"]

        method_path = func_info["function_parent"]
        file_path = bug_data["file"]["file_path"]

        project_name = adapter.map_project_name(bug_data["project_name"])
        defects4j_id = bug_data["defects4j_id"]
        checkout_path = os.path.join(base_checkout_dir, f"{project_name}_{defects4j_id}")
        success = adapter.checkout(project_name, defects4j_id, checkout_path)
        if not success:
            print(f"cannot checkout: {project_name}-{bug_id}")
            break

        java_file_path = os.path.join(checkout_path, file_path)
        if not os.path.exists(java_file_path):
            print(f"File not found after checkout: {bug_id}--{project_name}--{defects4j_id}--{java_file_path}")
            break

        try:
            with open(java_file_path, 'r', encoding='utf-8') as f:
                java_code = f.read()
            start_line, end_line = get_method_boundaries(java_code, method_path)
            if start_line and end_line:
                print(f"find start and end line!\n{bug_id}-{project_name}-{defects4j_id}-{java_file_path}")
                print(f"start_line: {start_line}, end_line: {end_line}")
                func_info["function_after_start_line_defects4j"] = start_line
                func_info["function_after_end_line_defects4j"] = end_line
                updated = True
            else:
                print(
                    f"cannot find the start_line and end_line: {bug_id}--{project_name}--{defects4j_id}--{java_file_path}")
                error_cases.append(f"{bug_id}-{project_name}-{defects4j_id}-{java_file_path}")
                continue

            if start_line != function_after_start_line or end_line != function_after_end_line:
                not_same_cases.append(f"{bug_id}-{project_name}-{defects4j_id}-{start_line}-{end_line}-{java_file_path}")
        except Exception as e:
            print(f"[Error] Failed to process bug {bug_id}: {e}")

    # print("✅ Done updating the JSON with `function_after_start_line_defects4j` and `function_after_end_line_defects4j`.")
    print(f"cannot find the start_line and end_line:\n{error_cases}")
    print(f"not equal with current:\n{not_same_cases}")


if __name__ == "__main__":
    # java_file_path = "Week.java"
    # with open(java_file_path, 'r', encoding='utf-8') as f:
    #     java_code = f.read()
    # method_path = "Week::Week( Date time , TimeZone zone)"
    # start_line, end_line = get_method_boundaries(java_code, method_path)
    # method_path = "AbstractCategoryItemRenderer::getLegendItems()"
    # method_path = "Week::Week( Date time , TimeZone zone)"
    # method_path = "HelpFormatter::appendOption( final StringBuffer buff , final Option option , final boolean required)"
    # method_path = "AnalyzePrototypeProperties::ProcessProperties::isPrototypePropertyAssign( Node assign)"
    # method_path = "Base64::encode( byte [ ] in , int inPos , int inAvail)"
    # method_path = "anyOtherEndTag( Token t , HtmlTreeBuilder tb)"
    # method_path = "TypeHandler::createValue( final String str , final Class<T> clazz)"
    # method_path = "SimpleType::_narrow( Class <?> subclass)"
    # method_path = "XmlTreeBuilder::popStackToClose( Token . EndTag endTag)"
    # method_path = "Variance::evaluate( final double [ ] values , final double [ ] weights , final double mean , final int begin , final int length)"
    # method_path = "MatrixUtil::multiply( int [ ] [ ] A , int [ ] [ ] B )"


    # Run this code in the defects4j docker container
    main()
