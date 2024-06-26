import json

# Sample JSON data, you would replace this with loading the JSON from a file or other source
json_data = """
{"kind": "Listing", "data": {"after": null, "children": [{"kind": "t3", "data": {"subreddit": "DebunkThis", "likes": null, "category": null, "created_utc": 1373450992.0, "pwls": null, "author_flair_richtext": [], "title": "\\"Cancer is a fungus\\" - this idea from the 60s is apparently receiving new attention. Please advise.", "spoiler": false, "author": "HyperSpaz", "saved": false, "suggested_sort": null, "mod_note": null, "mod_reason_by": null, "author_flair_template_id": null, "banned_at_utc": null, "link_flair_type": "text", "report_reasons": null, "approved_at_utc": null, "domain": "davidicke.com", "link_flair_css_class": null, "ups": 17, "upvote_ratio": 0.87, "approved_by": null, "rte_mode": "markdown", "num_comments": 10, "author_flair_text_color": null, "view_count": null, "post_categories": null, "author_flair_type": "text", "selftext": "", "removal_reason": null, "distinguished": null, "hidden": false, "stickied": false, "no_follow": false, "edited": false, "subreddit_name_prefixed": "r/DebunkThis", "pinned": false, "is_video": false, "id": "1hzz6y", "wls": null, "num_crossposts": 0, "score": 17, "author_flair_background_color": null, "quarantine": false, "is_original_content": false, "mod_reason_title": null, "subreddit_type": "public", "link_flair_text_color": "dark", "locked": false, "banned_by": null, "can_gild": false, "clicked": false, "url": "http://www.davidicke.com/articles/medicalhealth-mainmenu-37/29121", "subreddit_id": "t5_2rrbr", "num_reports": null, "permalink": "/r/DebunkThis/comments/1hzz6y/cancer_is_a_fungus_this_idea_from_the_60s_is/", "thumbnail_width": 70, "name": "t3_1hzz6y", "link_flair_richtext": [], "downs": 0, "send_replies": false, "media_only": false, "selftext_html": null, "hide_score": false, "secure_media_embed": {}, "thumbnail": "https://c.thumbs.redditmedia.com/UxdnkX4gYWclsA18.jpg", "is_self": false, "media": null, "media_embed": {}, "gilded": 0, "is_crosspostable": false, "subreddit_subscribers": 9404, "whitelist_status": null, "over_18": false, "archived": true, "secure_media": null, "created": 1373479792.0, "author_flair_text": null, "thumbnail_height": 53, "link_flair_text": null, "mod_reports": [], "user_reports": [], "author_flair_css_class": null, "contest_mode": false, "visited": false, "can_mod_post": false, "is_reddit_media_domain": false, "parent_whitelist_status": null}}], "modhash": "", "dist": 1, "before": null}}
"""

# Load JSON into a Python dictionary
data = json.loads(json_data)

# Access specific post information
post = data['data']['children'][0]['data']

# Print specific details about the post
print("Subreddit:", post['subreddit'])
print("Title:", post['title'])
print("Author:", post['author'])
print("URL:", post['url'])
