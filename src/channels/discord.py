import logging

import discord
from discord.ext import commands

from src.agent.core import AgentWithWorkflow
from src.config import DISCORD_TOKEN

logger = logging.getLogger("channels.discord")


async def start():
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN is not configured.")
        return

    intents = discord.Intents.default()
    intents.message_content = True  # 开启读取消息内容权限
    intents.members = True  # 开启成员相关权限

    bot = commands.Bot(command_prefix="!", intents=intents)
    agent_graph = AgentWithWorkflow()

    @bot.event
    async def on_ready():
        logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")

    @bot.event
    async def on_message(message):
        # 忽略 Bot 自己的消息，防止无限循环
        if message.author == bot.user:
            return

        # --- 场景 1：1对1 私聊 (DM) ---
        if isinstance(message.channel, discord.DMChannel):
            logger.info(f"收到私信 from: {message.author.id}")

            async with message.channel.typing():
                response = await agent_graph.chat(
                    user_id=str(message.author.id),
                    user_name=message.author.display_name,
                    user_input=message.content,
                    platform="discord",
                )

                # Discord requests must be under 2000 chars, so chunk if necessary
                if len(response) > 2000:
                    for i in range(0, len(response), 2000):
                        await message.channel.send(response[i : i + 2000])
                else:
                    await message.channel.send(response)
            return

        # --- 场景 2：Server 中被 @ 提到 ---
        if bot.user.mentioned_in(message):
            # 1. 在当前消息下创建一个 Thread
            thread_name = f"对话 - {message.author.display_name}"
            thread = await message.create_thread(
                name=thread_name,
                auto_archive_duration=60,  # 1小时后自动归档
            )

            # 2. 在 Thread 中 @ 对话发起者并回复
            # <@ID> 是 Discord 的提到格式
            await thread.send(f"你好 <@{message.author.id}>！我们在这里继续讨论。")

            # 3. 这里可以继续调用远程服务器获取内容
            # await thread.send("正在处理您的请求...")

        # 确保其他的 command 还能正常工作
        await bot.process_commands(message)

    await bot.start(DISCORD_TOKEN)
