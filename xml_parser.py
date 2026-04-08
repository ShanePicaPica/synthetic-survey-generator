"""
Decipher XML Parser - 從 XML 中提取問卷結構和跳題邏輯
"""
import re
from lxml import etree


def parse_decipher_xml(xml_content):
    """
    解析 Decipher XML，返回問卷結構字典
    """
    xml_content = re.sub(r'\bss:', 'ss_', xml_content)
    xml_content = re.sub(r'\bqa:', 'qa_', xml_content)
    xml_content = re.sub(r'\bkantar:', 'kantar_', xml_content)
    xml_content = re.sub(r'\bssc:', 'ssc_', xml_content)
    xml_content = re.sub(r'\bautosum:', 'autosum_', xml_content)
    xml_content = re.sub(r'\bcintrespondent:', 'cintrespondent_', xml_content)
    xml_content = re.sub(r'\bcintstatus:', 'cintstatus_', xml_content)
    xml_content = re.sub(r'\brecaptcha:', 'recaptcha_', xml_content)
    xml_content = re.sub(r'\bppg_getsession:', 'ppg_getsession_', xml_content)
    xml_content = re.sub(r'\bppg_sendoutcome:', 'ppg_sendoutcome_', xml_content)

    try:
        parser = etree.XMLParser(recover=True, encoding='utf-8')
        root = etree.fromstring(xml_content.encode('utf-8'), parser=parser)
    except Exception as e:
        return {"error": f"XML parsing failed: {str(e)}", "questions": []}

    questions = []
    _extract_questions(root, questions, parent_cond=None)

    return {
        "survey_name": root.get("alt", "Unknown Survey"),
        "questions": questions,
        "total_questions": len(questions)
    }


def _extract_questions(element, questions, parent_cond=None):
    """遞歸提取所有問題元素"""
    block_cond = None
    if element.tag == 'block':
        block_cond = element.get('cond')
        if block_cond == '0':
            return

    current_cond = block_cond or parent_cond

    for child in element:
        tag = child.tag

        if tag in ('radio', 'checkbox', 'select', 'number', 'float', 'text', 'textarea'):
            q = _parse_question(child, current_cond)
            if q:
                questions.append(q)

        elif tag in ('block', 'loop'):
            loop_cond = child.get('cond')
            if loop_cond == '0':
                continue
            effective_cond = loop_cond if loop_cond else current_cond
            _extract_questions(child, questions, parent_cond=effective_cond)


def _parse_question(element, parent_cond=None):
    """解析單個問題元素"""
    label = element.get('label', '')

    skip_prefixes = (
        'h_', 'hid_', 'Info', 'PIICN', 'chk', 'ln1', 'ln2',
        'changerecord', 'userAgent', 'vpanels', 'vtest',
        'pagetime_', 'QC_', 'check_', 'PPG_', 'sample'
    )
    if any(label.startswith(p) for p in skip_prefixes):
        return None

    where = element.get('where', '')
    if 'notdp' in where and 'survey' not in where.replace('notdp', ''):
        return None

    cond = element.get('cond', '')
    if cond == '0':
        return None

    tag = element.tag
    title_elem = element.find('title')
    title = ''
    if title_elem is not None:
        title = ''.join(title_elem.itertext()).strip()
        title = re.sub(r'\[pipe:\w+\]', '[PIPED]', title)
        title = re.sub(r'\$\{[^}]+\}', '[DYNAMIC]', title)

    rows = []
    for row in element.findall('.//row'):
        row_label = row.get('label', '')
        row_val = row.get('value', '')
        row_text = ''.join(row.itertext()).strip()
        row_cond = row.get('cond', '')
        exclusive = row.get('exclusive', '') == '1'
        has_open = row.get('open', '') == '1'
        rows.append({
            "label": row_label,
            "value": row_val,
            "text": row_text,
            "cond": row_cond,
            "exclusive": exclusive,
            "has_open": has_open
        })

    cols = []
    for col in element.findall('.//col'):
        col_label = col.get('label', '')
        col_val = col.get('value', '')
        col_text = ''.join(col.itertext()).strip()
        cols.append({
            "label": col_label,
            "value": col_val,
            "text": col_text
        })

    choices = []
    for ch in element.findall('.//choice'):
        choices.append({
            "label": ch.get('label', ''),
            "text": ''.join(ch.itertext()).strip()
        })

    q_type = _determine_question_type(tag, element, rows, cols, choices)

    effective_cond = cond if cond else parent_cond

    validate_elem = element.find('validate')
    validate_text = ''
    if validate_elem is not None:
        validate_text = ''.join(validate_elem.itertext()).strip()

    atleast = element.get('atleast', '')
    atmost = element.get('atmost', '')
    optional = element.get('optional', '')
    shuffle = element.get('shuffle', '')
    verify = element.get('verify', '')
    size = element.get('size', '')

    return {
        "label": label,
        "type": q_type,
        "tag": tag,
        "title": title[:200] if title else '',
        "cond": effective_cond or '',
        "rows": rows,
        "cols": cols,
        "choices": choices,
        "atleast": atleast,
        "atmost": atmost,
        "optional": optional,
        "shuffle": shuffle,
        "verify": verify,
        "validate": validate_text,
        "size": size
    }


def _determine_question_type(tag, element, rows, cols, choices):
    """判斷具體問題類型"""
    if tag == 'radio' and cols:
        return 'grid_radio'
    elif tag == 'radio':
        return 'single'
    elif tag == 'checkbox':
        return 'multi'
    elif tag == 'select' and choices:
        return 'ranking'
    elif tag == 'number':
        grouping = element.get('grouping', '')
        if grouping == 'cols' or element.find('.//col') is not None:
            return 'numeric_grid'
        elif len(rows) > 1:
            return 'numeric_multi'
        else:
            return 'numeric'
    elif tag == 'float':
        return 'numeric'
    elif tag == 'text':
        if len(rows) > 0:
            return 'open_multi'
        return 'open_end'
    elif tag == 'textarea':
        return 'open_end'
    return 'unknown'


def get_question_summary(parsed):
    """生成問卷結構摘要"""
    type_counts = {}
    for q in parsed['questions']:
        t = q['type']
        type_counts[t] = type_counts.get(t, 0) + 1

    conditional_qs = sum(1 for q in parsed['questions'] if q['cond'])

    return {
        "survey_name": parsed['survey_name'],
        "total_questions": parsed['total_questions'],
        "type_breakdown": type_counts,
        "conditional_questions": conditional_qs,
        "questions_list": [
            {
                "label": q['label'],
                "type": q['type'],
                "title": q['title'][:100],
                "cond": q['cond'][:100] if q['cond'] else 'Always shown',
                "options_count": len(q['rows'])
            }
            for q in parsed['questions']
        ]
    }
