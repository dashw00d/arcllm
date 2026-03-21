#!/usr/bin/env python3
"""Henry — Discord bot backed by local Qwen3-32B via llama-server."""

import logging

import discord
from discord import app_commands

from config import DISCORD_TOKEN
from history import HistoryManager
from inference import chat
from prompts import DISCORD_SYSTEM_PROMPT
from rate_limiter import RateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("bot")

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)
history = HistoryManager()
limiter = RateLimiter()


@bot.event
async def on_ready():
    await tree.sync()
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # henrys-hotdog-stand: respond to everything. All other channels: mention only.
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_hotdog_stand = message.channel.id == 1482957506002030592
    is_mentioned = bot.user in message.mentions
    if not is_dm and not is_hotdog_stand and not is_mentioned:
        return

    if not limiter.check(message.author.id):
        await message.add_reaction("\u23f3")  # hourglass
        return

    # Strip the bot mention from the message content
    content = message.content.replace(f"<@{bot.user.id}>", "").strip()
    if not content:
        return

    channel_id = message.channel.id
    history.add_user(channel_id, message.author.display_name, content)

    messages = [
        {"role": "system", "content": DISCORD_SYSTEM_PROMPT},
        *history.get(channel_id),
    ]

    try:
        async with message.channel.typing():
            reply = await chat(messages)
        history.add_assistant(channel_id, reply)
        for chunk in _split_message(reply):
            await message.reply(chunk, mention_author=False)
    except discord.Forbidden:
        log.warning("Missing permissions in channel %s", channel_id)
    except Exception:
        log.exception("Error handling message in channel %s", channel_id)


def _split_message(text: str, limit: int = 1990) -> list[str]:
    """Split text into chunks that fit Discord's message limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split on newline
        idx = text.rfind("\n", 0, limit)
        if idx == -1:
            idx = limit
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks


# ── Slash Commands ────────────────────────────────────────────────────────

@tree.command(name="reset", description="Clear conversation history for this channel")
async def cmd_reset(interaction: discord.Interaction):
    history.clear(interaction.channel_id)
    await interaction.response.send_message("History cleared.", ephemeral=True)


@tree.command(name="think", description="Ask a question with extended thinking enabled")
@app_commands.describe(question="Your question")
async def cmd_think(interaction: discord.Interaction, question: str):
    if not limiter.check(interaction.user.id):
        await interaction.response.send_message("Slow down!", ephemeral=True)
        return

    await interaction.response.defer()
    channel_id = interaction.channel_id
    history.add_user(channel_id, interaction.user.display_name, question)

    messages = [
        {"role": "system", "content": DISCORD_SYSTEM_PROMPT},
        *history.get(channel_id),
    ]

    try:
        reply = await chat(messages, thinking=True)
        history.add_assistant(channel_id, reply)
        for chunk in _split_message(reply):
            await interaction.followup.send(chunk)
    except Exception:
        log.exception("Error in /think")
        await interaction.followup.send("Something went wrong.")


@tree.command(name="status", description="Show inference server status")
async def cmd_status(interaction: discord.Interaction):
    from inference import client
    try:
        models = await client.models.list()
        model_list = ", ".join(m.id for m in models.data[:5])
        await interaction.response.send_message(
            f"Server up. Models: {model_list}", ephemeral=True
        )
    except Exception as e:
        await interaction.response.send_message(f"Server error: {e}", ephemeral=True)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
