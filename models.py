from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from typing import Literal


class VoiceInfo(BaseModel):
    ShortName: str
    Gender: str
    Locale: str
    FriendlyName: str
    VoiceType: str | None = None


class VoiceCatalog(BaseModel):
    voices: list[VoiceInfo]


class SpeakRequest(BaseModel):
    input_type: Literal["text", "url"] = Field(default="text")
    text: str | None = Field(default=None, max_length=50000)
    url: HttpUrl | None = None
    voice: str = Field(default="en-US-AndrewNeural")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
        return v


class SpeakResponse(BaseModel):
    speak_id: str
    text_length: int
    num_segments: int
    voice: str


class StatusResponse(BaseModel):
    state: Literal["idle", "synthesizing", "playing"]
    speak_id: str | None = None
    segments_total: int = 0
    segments_done: int = 0
    elapsed_ms: int = 0


class ErrorResponse(BaseModel):
    detail: str


class BTDevice(BaseModel):
    mac: str
    name: str
    paired: bool = False
    connected: bool = False


class BTHistoryEntry(BaseModel):
    mac: str
    name: str
    last_connected: int


class BTStatusResponse(BaseModel):
    connected: bool
    device: BTDevice | None = None
    system_default: BTDevice | None = None
    history: list[BTHistoryEntry] = []


class BTScanResponse(BaseModel):
    devices: list[BTDevice]


class BTConnectRequest(BaseModel):
    mac: str


class BTConnectResponse(BaseModel):
    ok: bool
    connected: bool
    device: BTDevice | None = None


class SpeakerState(BaseModel):
    mac: str
    name: str
    volume: int | None = None
    muted: bool | None = None
    battery: int | None = None


class SpeakerVolumeRequest(BaseModel):
    volume: int = Field(ge=0, le=127)


class SpeakerMuteRequest(BaseModel):
    muted: bool


class RoleInfo(BaseModel):
    role: str
    voice: str


class SegmentInfo(BaseModel):
    role: str
    start_word: int
    end_word: int


class StoryPlan(BaseModel):
    story_name: str
    roles: list[RoleInfo]
    segments: list[SegmentInfo]
    llm_response: dict | None = None


class StoryInfo(BaseModel):
    name: str
    roles: list[str]
    segment_count: int
    created: float


class StoryListResponse(BaseModel):
    stories: list[StoryInfo]


class StoryAnalyzeRequest(BaseModel):
    url: HttpUrl | None = None
    text: str | None = None

    @model_validator(mode="after")
    def check_one_source(self):
        if (self.url is None) == (self.text is None):
            raise ValueError("Provide exactly one of url or text")
        return self


class StoryGenerateRequest(BaseModel):
    url: HttpUrl | None = None
    text: str | None = None
    story_name: str
    plan: StoryPlan

    @model_validator(mode="after")
    def check_one_source(self):
        if (self.url is None) == (self.text is None):
            raise ValueError("Provide exactly one of url or text")
        return self


class StoryTaskStatus(BaseModel):
    task_id: str
    state: Literal["running", "done", "failed", "cancelled"]
    total: int = 0
    done: int = 0
    current_role: str | None = None
    error: str | None = None
