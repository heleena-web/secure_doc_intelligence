import re
from typing import Any, Optional
from app.models.document import DocumentType, ParsedBlock, ParsedDocument


def extract_rich_text(rich_text_list: list[dict[str, Any]]) -> str:
    """Concatenates plain_text elements from a Notion rich_text array."""
    if not isinstance(rich_text_list, list):
        return ""
    parts = []
    for item in rich_text_list:
        if isinstance(item, dict):
            text = item.get("plain_text", "")
            parts.append(text)
    return "".join(parts)


def extract_page_title(page_data: dict[str, Any]) -> str:
    """Extracts page title from Notion page properties dictionary."""
    properties = page_data.get("properties", {})
    if not isinstance(properties, dict):
        return "Untitled Notion Page"

    for prop_name, prop_val in properties.items():
        if isinstance(prop_val, dict) and prop_val.get("type") == "title":
            title_list = prop_val.get("title", [])
            title = extract_rich_text(title_list).strip()
            if title:
                return title

    # Fallback to id if title cannot be resolved
    page_id = page_data.get("id", "Page")
    return f"Notion Page {page_id[:8]}"


def extract_page_properties_summary(page_data: dict[str, Any]) -> str:
    """Extracts database/page properties (tags, status, select, dates, notes) into searchable text."""
    properties = page_data.get("properties", {})
    if not isinstance(properties, dict):
        return ""
    lines = []
    for prop_name, prop_val in properties.items():
        if not isinstance(prop_val, dict):
            continue
        p_type = prop_val.get("type")
        if p_type == "title":
            continue  # Title is already page heading
        elif p_type == "rich_text":
            text = extract_rich_text(prop_val.get("rich_text", [])).strip()
            if text:
                lines.append(f"{prop_name}: {text}")
        elif p_type == "select":
            sel = prop_val.get("select")
            if sel and isinstance(sel, dict) and sel.get("name"):
                lines.append(f"{prop_name}: {sel['name']}")
        elif p_type == "multi_select":
            items = prop_val.get("multi_select", [])
            names = [it["name"] for it in items if isinstance(it, dict) and it.get("name")]
            if names:
                lines.append(f"{prop_name}: {', '.join(names)}")
        elif p_type == "status":
            st = prop_val.get("status")
            if st and isinstance(st, dict) and st.get("name"):
                lines.append(f"{prop_name}: {st['name']}")
        elif p_type == "checkbox":
            lines.append(f"{prop_name}: {'Checked' if prop_val.get('checkbox') else 'Unchecked'}")
        elif p_type == "date":
            d = prop_val.get("date")
            if d and isinstance(d, dict) and d.get("start"):
                lines.append(f"{prop_name}: {d['start']}")
        elif p_type == "url":
            u = prop_val.get("url")
            if u:
                lines.append(f"{prop_name}: {u}")
        elif p_type == "number":
            n = prop_val.get("number")
            if n is not None:
                lines.append(f"{prop_name}: {n}")
    return "\n".join(lines)


class NotionParser:
    """Transforms raw Notion blocks and page metadata into a canonical ParsedDocument.
    Preserves section hierarchy, tables, lists, code, callouts, columns, and URLs.
    Gracefully ignores unsupported blocks without errors.
    """

    def parse_page(
        self,
        page_data: dict[str, Any],
        blocks: list[dict[str, Any]],
    ) -> ParsedDocument:
        page_id = page_data.get("id", "")
        title = extract_page_title(page_data)
        url = page_data.get("url", f"https://notion.so/{page_id.replace('-', '')}")
        last_edited_time = page_data.get("last_edited_time", "")

        parsed_blocks: list[ParsedBlock] = []
        current_h1: Optional[str] = None
        current_h2: Optional[str] = None
        current_h3: Optional[str] = None

        def current_heading_path() -> Optional[str]:
            parts = [h for h in (current_h1, current_h2, current_h3) if h]
            return " > ".join(parts) if parts else None

        # 1. Extract metadata properties (for Notion database rows / tagged pages)
        prop_summary = extract_page_properties_summary(page_data)
        if prop_summary:
            parsed_blocks.append(
                ParsedBlock(
                    text=f"Page Properties:\n{prop_summary}",
                    heading_path=None,
                    metadata={"block_type": "page_properties"},
                )
            )

        # 2. Recursive block parser
        def parse_block_list(block_list: list[dict[str, Any]]) -> None:
            nonlocal current_h1, current_h2, current_h3
            for block in block_list:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type", "")
                block_data = block.get(block_type, {})
                if not isinstance(block_data, dict):
                    block_data = {}

                # Heading 1
                if block_type == "heading_1":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    if text:
                        current_h1 = text
                        current_h2 = None
                        current_h3 = None
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"# {text}",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Heading 2
                elif block_type == "heading_2":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    if text:
                        current_h2 = text
                        current_h3 = None
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"## {text}",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Heading 3
                elif block_type == "heading_3":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    if text:
                        current_h3 = text
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"### {text}",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Paragraph
                elif block_type == "paragraph":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    if text:
                        parsed_blocks.append(
                            ParsedBlock(
                                text=text,
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Bulleted List
                elif block_type == "bulleted_list_item":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    if text:
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"• {text}",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Numbered List
                elif block_type == "numbered_list_item":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    if text:
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"1. {text}",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # To-Do Checkbox
                elif block_type == "to_do":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    checked = block_data.get("checked", False)
                    prefix = "[x]" if checked else "[ ]"
                    if text:
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"{prefix} {text}",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id"), "checked": checked},
                            )
                        )

                # Quote
                elif block_type == "quote":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    if text:
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"> {text}",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Code
                elif block_type == "code":
                    text = extract_rich_text(block_data.get("rich_text", []))
                    lang = block_data.get("language", "text")
                    if text.strip():
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"```{lang}\n{text}\n```",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id"), "language": lang},
                            )
                        )

                # Callout
                elif block_type == "callout":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    icon_data = block_data.get("icon", {})
                    icon_emoji = icon_data.get("emoji", "ℹ️") if isinstance(icon_data, dict) else "ℹ️"
                    if text:
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"[{icon_emoji}] {text}",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Table with child table_row blocks
                elif block_type == "table":
                    child_rows = block.get("child_blocks", [])
                    table_lines = []
                    for row_block in child_rows:
                        if row_block.get("type") == "table_row":
                            cells = row_block.get("table_row", {}).get("cells", [])
                            row_texts = [extract_rich_text(cell).strip() for cell in cells]
                            table_lines.append("| " + " | ".join(row_texts) + " |")

                    if table_lines:
                        table_text = "\n".join(table_lines)
                        parsed_blocks.append(
                            ParsedBlock(
                                text=table_text,
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Toggle
                elif block_type == "toggle":
                    text = extract_rich_text(block_data.get("rich_text", [])).strip()
                    if text:
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"▼ {text}",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Column List / Column / Synced Block (container blocks)
                elif block_type in ("column_list", "column", "synced_block"):
                    pass  # children processed below

                # Child page & database reference
                elif block_type in ("child_page", "child_database"):
                    c_title = block_data.get("title", "")
                    if c_title:
                        label = "Sub-page" if block_type == "child_page" else "Database"
                        parsed_blocks.append(
                            ParsedBlock(
                                text=f"[{label}: {c_title}]",
                                heading_path=current_heading_path(),
                                metadata={"block_type": block_type, "block_id": block.get("id")},
                            )
                        )

                # Divider
                elif block_type == "divider":
                    parsed_blocks.append(
                        ParsedBlock(
                            text="---",
                            heading_path=current_heading_path(),
                            metadata={"block_type": block_type, "block_id": block.get("id")},
                        )
                    )

                # Fallback for other block types with rich_text
                else:
                    if "rich_text" in block_data:
                        fallback_text = extract_rich_text(block_data.get("rich_text", [])).strip()
                        if fallback_text:
                            parsed_blocks.append(
                                ParsedBlock(
                                    text=fallback_text,
                                    heading_path=current_heading_path(),
                                    metadata={"block_type": block_type, "block_id": block.get("id")},
                                )
                            )

                # Recursively process children of container blocks (columns, toggles, callouts, lists)
                if block_type != "table" and block.get("child_blocks"):
                    parse_block_list(block["child_blocks"])

        parse_block_list(blocks)

        # If page had title but no blocks, include title as block
        if not parsed_blocks and title:
            parsed_blocks.append(
                ParsedBlock(
                    text=title,
                    heading_path=title,
                    metadata={"source": "title_only"},
                )
            )

        char_count = sum(len(b.text) for b in parsed_blocks)

        return ParsedDocument(
            filename=title,
            file_type=DocumentType.NOTION.value,
            char_count=char_count,
            blocks=parsed_blocks,
            page_count=1,
            metadata={
                "source_type": "notion",
                "notion_page_id": page_id,
                "url": url,
                "last_edited_time": last_edited_time,
            },
        )
