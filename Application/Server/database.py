#  @bekbrace
#  FARMSTACK Tutorial - Sunday 13.06.2021

import motor.motor_asyncio
from model import Comment

client = motor.motor_asyncio.AsyncIOMotorClient('mongodb+srv://mani:rf2G8VFfChrnf1pS@cluster0.ncpvelk.mongodb.net/')
database = client.SocialMedia
collection = database.comments

async def fetch_comments(label):
    comments = []
    try:
        cursor = collection.find({"label": label})
        if cursor:
            async for document in cursor:
                comments.append(Comment(**document))
            return comments
        else:
            return False
    except KeyError:
            # Handle documents with missing or incorrect fields
            print(f"Skipping document with missing or incorrect fields: {document}")
            pass

async def fetch_all_comments():
    comments = []
    cursor = collection.find({})
    async for document in cursor:
        comments.append(Comment(**document))
    return comments

async def create_comment(comment):
    document = comment
    await collection.insert_one(document)
    return True


# async def update_comment(title, desc):
#     await collection.update_one({"title": title}, {"$set": {"description": desc}})
#     document = await collection.find_one({"title": title})
#     return document

# async def remove_comment(title):
#     await collection.delete_one({"title": title})
#     return True