import json
import sys


def load_bug_data(filepath: str) -> dict:
    """안전하게 JSON 파일을 로드합니다."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 에러: 입력 파일 '{filepath}'을(를) 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ 에러: '{filepath}' 파일이 올바른 JSON 형식이 아닙니다.", file=sys.stderr)
        sys.exit(1)


def restructure_all_and_save(bugs_data: dict, output_filepath: str):
    """
    모든 버그 데이터를 순회하며, 사용자가 요청한 새로운 구조로 재구성하여 JSON 파일에 저장합니다.
    """
    if not bugs_data:
        print("처리할 버그 데이터가 없습니다.")
        return

    # 재구성된 최종 결과를 저장할 딕셔너리
    restructured_data = {}

    print(f"총 {len(bugs_data)}개의 모든 버그 데이터에 대한 재구성 작업을 시작합니다...")

    # 모든 키(bug_id)와 값(item)을 순차적으로 반복
    for bug_id, item in bugs_data.items():
        # --- 요청하신 형식에 맞춰 데이터 추출 및 재구성 ---

        # 각 객체에서 정보를 가져오기 (get을 사용하여 키가 없어도 오류 방지)
        file_info = item.get("file", {})
        commit_info = item.get("commit", {})
        function_info = item.get("function", {})

        # 새로운 딕셔너리 생성
        new_structure = {
            "bug_id": bug_id,
            "bugsinpy_id":item.get("bugsinpy_id", "N/A"),
            "buggy_line_location":item.get("buggy_line_location", "N/A"),
            "project_name": item.get("project_name", "N/A"),
            "project_url": item.get("project_url", "N/A"),
            "file_name": file_info.get("file_name", "N/A"),
            "file_path": file_info.get("file_path", "N/A"),
            "function_name": function_info.get("function_name", "N/A"),
            "buggy_line_content": item.get("buggy_line_content", "").strip(),
            "buggy_line_context": item.get("buggy_line_context", ""),  # 원본에 없으므로 기본값 ""
            "commit": commit_info,  # commit 객체 전체를 그대로 복사
            "function": function_info  # function 객체 전체를 그대로 복사
        }

        # 최종 결과 딕셔너리에 추가
        restructured_data[bug_id] = new_structure
        print(f"  - '{bug_id}' 키의 데이터 재구성 완료.")

    # 최종적으로 재구성된 딕셔너리를 JSON 파일로 저장
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(restructured_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 성공: 재구성된 전체 데이터가 '{output_filepath}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 실패: 파일 저장 중 오류가 발생했습니다. {e}")


if __name__ == "__main__":
    input_file = "./defects4j_bugs_meta_data.json"
    output_file = "./Results/1.DataSeterAgentResult_Java.json"

    # 1. 원본 데이터 로드
    all_bugs_data = load_bug_data(input_file)

    # 2. 모든 데이터를 순회하며 새로운 구조로 만들어 파일로 저장
    restructure_all_and_save(all_bugs_data, output_file)