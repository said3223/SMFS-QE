//! SMFS-QE — Semantic Memory Filesystem (Quantum-Enhanced)
//!
//! Дизайн-цели v0.2:
//! - Блочная модель 2–32 KiB с метаданными и семантическими тегами.
//! - Порядковая адресация (IndexMap) для детерминированной реконструкции reasoning-цепочек.
//! - Диск-слой (StorageBackend) + простая InMemory/FS реализация.
//! - Заглушки SAQ (квантование) и EIME (шифрование) как сменные бэкенды.
//! - Хелперы: поиск по тегам, чтение/запись диапазонов, контроль целостности (xxhash3).
//!
//! Интеграция с SRIS:
//! - SRK/RIU обращаются к `SMFSQEManager` через стабильные трейт-интерфейсы.
//! - Semantic Scheduler может использовать `semantic_tag`/`t_unit_id` и `iter_ordered()`.
//! - SAQ/EIME подключаются фичами, не ломая API.

use indexmap::IndexMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use xxhash_rust::xxh3::xxh3_64;

/// Базовые размеры блоков (под твой диапазон SAQ-квантов).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(not(feature = "serde"), derive(Clone, Copy, PartialEq, Eq, Debug))]
pub enum BlockSize {
    B2KB = 2048,
    B4KB = 4096,
    B8KB = 8192,
    B16KB = 16384,
    B32KB = 32768,
}

impl BlockSize {
    #[inline]
    pub fn as_usize(self) -> usize {
        self as usize
    }
}

/// Метаданные блока — отделены от данных для удобства сериализации/SAQ.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryBlockMeta {
    pub id: u64,
    pub size: BlockSize,
    pub is_used: bool,
    /// Семантическая метка (например: "episode:inspection", "t:vision", и т.п.)
    pub semantic_tag: Option<String>,
    /// Идентификатор тессеракт-юнита (атом смысла), если известен.
    pub t_unit_id: Option<u128>,
    /// Контрольная сумма полезных данных (после SAQ/шифрования — по выбранной стадии).
    pub checksum: u64,
    /// Версия формата блока (на будущее для миграций).
    pub version: u16,
}

impl Default for MemoryBlockMeta {
    fn default() -> Self {
        Self {
            id: 0,
            size: BlockSize::B4KB,
            is_used: false,
            semantic_tag: None,
            t_unit_id: None,
            checksum: 0,
            version: 1,
        }
    }
}

/// Сам блок: данные + метаданные.
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub meta: MemoryBlockMeta,
    pub data: Vec<u8>,
}

impl MemoryBlock {
    pub fn new(id: u64, size: BlockSize) -> Self {
        let mut b = Self {
            meta: MemoryBlockMeta {
                id,
                size,
                is_used: false,
                ..Default::default()
            },
            data: vec![0; size.as_usize()],
        };
        b.recompute_checksum();
        b
    }

    #[inline]
    pub fn view(&self) -> &[u8] { &self.data }
    #[inline]
    pub fn view_mut(&mut self) -> &mut [u8] { &mut self.data }

    pub fn mark_used(&mut self, used: bool) { self.meta.is_used = used; }

    pub fn set_tag(&mut self, tag: Option<String>) { self.meta.semantic_tag = tag; }

    pub fn set_t_unit(&mut self, t: Option<u128>) { self.meta.t_unit_id = t; }

    pub fn recompute_checksum(&mut self) {
        self.meta.checksum = xxh3_64(&self.data);
    }

    pub fn verify_checksum(&self) -> bool {
        xxh3_64(&self.data) == self.meta.checksum
    }
}

/* --------------------------- Ошибки уровня SMFS --------------------------- */

#[derive(Error, Debug)]
pub enum SmfsError {
    #[error("Block not found: {0}")]
    BlockNotFound(u64),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Out of range write/read")]
    Range,

    #[error("Integrity check failed for block {0}")]
    Integrity(u64),

    #[error("Backend error: {0}")]
    Backend(String),
}

pub type SmfsResult<T> = Result<T, SmfsError>;

/* ------------------------ SAQ: профили квантования ------------------------ */

/// Уровень точности для SAQ. Реальная логика позже; пока — заглушки.
#[derive(Debug, Clone, Copy)]
pub enum PrecisionTier {
    Lossless,   // без потерь
    P10,        // 10-бит условной точности (пример)
    P8,         // 8-бит
    P6,         // 6-бит и т.д.
}

/// Профиль SAQ: куда планируем квантизировать данные.
#[derive(Debug, Clone)]
pub struct SaqProfile {
    pub tier: PrecisionTier,
    pub preserve_peaks: bool,
}

impl Default for SaqProfile {
    fn default() -> Self {
        Self { tier: PrecisionTier::Lossless, preserve_peaks: true }
    }
}

/// Простейший SAQ-бэкенд (заглушка): ничего не меняет (Lossless) или «усредняет» биты.
pub trait SaqBackend: Send + Sync {
    fn quantize(&self, input: &[u8], profile: &SaqProfile) -> Vec<u8>;
    fn dequantize(&self, input: &[u8], profile: &SaqProfile) -> Vec<u8>;
}

/// Бэкенд по умолчанию — no-op для Lossless и примитивный даунсемплинг для демонстрации.
pub struct SaqNaive;

impl SaqBackend for SaqNaive {
    fn quantize(&self, input: &[u8], profile: &SaqProfile) -> Vec<u8> {
        match profile.tier {
            PrecisionTier::Lossless => input.to_vec(),
            _ => {
                // Простейшая «квантизация»: грубое усечение младших бит (демо).
                input.iter().map(|b| b & 0b1111_0000).collect()
            }
        }
    }
    fn dequantize(&self, input: &[u8], _profile: &SaqProfile) -> Vec<u8> {
        input.to_vec() // В реальности тут аппроксимация.
    }
}

/* ---------------------- EIME: шифрование/целостность ---------------------- */

/// Шифрующий бэкенд. Можно подвесить AEAD (ChaCha20Poly1305) фичей.
pub trait CipherBackend: Send + Sync {
    fn encrypt(&self, plaintext: &[u8]) -> SmfsResult<Vec<u8>>;
    fn decrypt(&self, ciphertext: &[u8]) -> SmfsResult<Vec<u8>>;
}

/// По умолчанию — просто echo (безопасность отключена, только для dev).
pub struct CipherNone;
impl CipherBackend for CipherNone {
    fn encrypt(&self, p: &[u8]) -> SmfsResult<Vec<u8>> { Ok(p.to_vec()) }
    fn decrypt(&self, c: &[u8]) -> SmfsResult<Vec<u8>> { Ok(c.to_vec()) }
}

/* --------------------------- Диск-/хранилище-слой ------------------------- */

/// Универсальный интерфейс хранения блоков.
pub trait StorageBackend: Send + Sync {
    fn write_block(&mut self, blk: &MemoryBlock) -> SmfsResult<()>;
    fn read_block(&mut self, id: u64, size: BlockSize) -> SmfsResult<MemoryBlock>;
    fn remove_block(&mut self, id: u64) -> SmfsResult<()>;
}

/// In-memory хранилище (удобно для тестов/RIU-симуляций).
pub struct InMemoryStorage {
    blobs: IndexMap<u64, Vec<u8>>,
}

impl InMemoryStorage {
    pub fn new() -> Self { Self { blobs: IndexMap::new() } }
}

impl StorageBackend for InMemoryStorage {
    fn write_block(&mut self, blk: &MemoryBlock) -> SmfsResult<()> {
        self.blobs.insert(blk.meta.id, blk.data.clone());
        Ok(())
    }

    fn read_block(&mut self, id: u64, size: BlockSize) -> SmfsResult<MemoryBlock> {
        let data = self.blobs.get(&id).ok_or(SmfsError::BlockNotFound(id))?;
        let mut b = MemoryBlock::new(id, size);
        b.data[..data.len().min(b.data.len())].copy_from_slice(&data[..data.len().min(b.data.len())]);
        b.mark_used(true);
        b.recompute_checksum();
        Ok(b)
    }

    fn remove_block(&mut self, id: u64) -> SmfsResult<()> {
        self.blobs.swap_remove(&id);
        Ok(())
    }
}

/// Файловое хранилище: по одному файлу на блок (простая, но понятная стратегия).
pub struct FsStorage {
    root: PathBuf,
}

impl FsStorage {
    pub fn new<P: AsRef<Path>>(root: P) -> SmfsResult<Self> {
        std::fs::create_dir_all(&root)?;
        Ok(Self { root: root.as_ref().to_path_buf() })
    }

    fn path_for(&self, id: u64) -> PathBuf {
        self.root.join(format!("{id:016x}.blk"))
    }
}

impl StorageBackend for FsStorage {
    fn write_block(&mut self, blk: &MemoryBlock) -> SmfsResult<()> {
        let mut f = OpenOptions::new().create(true).write(true).truncate(true).open(self.path_for(blk.meta.id))?;
        f.write_all(&blk.data)?;
        Ok(())
    }

    fn read_block(&mut self, id: u64, size: BlockSize) -> SmfsResult<MemoryBlock> {
        let mut f = File::open(self.path_for(id))?;
        let mut buf = vec![0u8; size.as_usize()];
        let n = f.read(&mut buf)?;
        if n < buf.len() {
            // добиваем нулями
            buf[n..].fill(0);
        }
        let mut b = MemoryBlock::new(id, size);
        b.data.copy_from_slice(&buf);
        b.mark_used(true);
        b.recompute_checksum();
        Ok(b)
    }

    fn remove_block(&mut self, id: u64) -> SmfsResult<()> {
        let p = self.path_for(id);
        if p.exists() { std::fs::remove_file(p)?; }
        Ok(())
    }
}

/* ----------------------------- Менеджер SMFS ------------------------------ */

pub struct SMFSQEManager<B: StorageBackend, S: SaqBackend, C: CipherBackend> {
    pub blocks: IndexMap<u64, MemoryBlockMeta>, // порядок = порядок аллокаций/коммитов
    next_id: u64,
    pub backend: B,
    pub saq: S,
    pub cipher: C,
    pub default_profile: SaqProfile,
}

impl<B: StorageBackend, S: SaqBackend, C: CipherBackend> SMFSQEManager<B, S, C> {
    pub fn new(backend: B, saq: S, cipher: C) -> Self {
        Self {
            blocks: IndexMap::new(),
            next_id: 0,
            backend,
            saq,
            cipher,
            default_profile: SaqProfile::default(),
        }
    }

    /// Аллокация нового блока и запись (пока пустой) на бекенд.
    pub fn allocate_block(&mut self, size: BlockSize, tag: Option<String>, t_unit: Option<u128>) -> SmfsResult<u64> {
        let id = self.next_id;
        let mut block = MemoryBlock::new(id, size);
        block.mark_used(true);
        block.set_tag(tag);
        block.set_t_unit(t_unit);
        block.recompute_checksum();
        self.backend.write_block(&block)?;
        self.blocks.insert(id, block.meta.clone());
        self.next_id += 1;
        Ok(id)
    }

    /// Деаллокация (данные на бэкенде удаляются, метаданные помечаются как свободные).
    pub fn deallocate_block(&mut self, id: u64) -> SmfsResult<()> {
        let meta = self.blocks.get_mut(&id).ok_or(SmfsError::BlockNotFound(id))?;
        meta.is_used = false;
        self.backend.remove_block(id)?;
        Ok(())
    }

    /// Прочитать блок целиком (с верификацией целостности).
    pub fn read_block(&mut self, id: u64) -> SmfsResult<MemoryBlock> {
        let meta = self.blocks.get(&id).ok_or(SmfsError::BlockNotFound(id))?.clone();
        let mut block = self.backend.read_block(id, meta.size)?;
        // (Если включены SAQ/EIME — здесь обычно выполняется decrypt+dequantize.)
        if !block.verify_checksum() {
            return Err(SmfsError::Integrity(id));
        }
        Ok(block)
    }

    /// Записать полезные данные в диапазон блока (in-place). Без смены размера.
    pub fn write_range(&mut self, id: u64, offset: usize, data: &[u8]) -> SmfsResult<()> {
        let meta = self.blocks.get(&id).ok_or(SmfsError::BlockNotFound(id))?.clone();
        let mut block = self.backend.read_block(id, meta.size)?;
        let end = offset.checked_add(data.len()).ok_or(SmfsError::Range)?;
        if end > block.data.len() { return Err(SmfsError::Range); }
        block.data[offset..end].copy_from_slice(data);
        block.recompute_checksum();
        self.backend.write_block(&block)?;
        self.blocks.insert(id, block.meta); // обновили checksum
        Ok(())
    }

    /// Прочитать диапазон.
    pub fn read_range(&mut self, id: u64, offset: usize, len: usize) -> SmfsResult<Vec<u8>> {
        let meta = self.blocks.get(&id).ok_or(SmfsError::BlockNotFound(id))?.clone();
        let block = self.backend.read_block(id, meta.size)?;
        let end = offset.checked_add(len).ok_or(SmfsError::Range)?;
        if end > block.data.len() { return Err(SmfsError::Range); }
        Ok(block.data[offset..end].to_vec())
    }

    /// Поиск по семантической метке.
    pub fn find_by_tag<'a>(&'a self, tag: &str) -> impl Iterator<Item = (&u64, &MemoryBlockMeta)> + 'a {
        self.blocks.iter().filter(move |(_, m)| m.semantic_tag.as_deref() == Some(tag))
    }

    /// Итерация в детерминированном порядке (по вставке).
    pub fn iter_ordered(&self) -> impl Iterator<Item = (&u64, &MemoryBlockMeta)> {
        self.blocks.iter()
    }
}

/* ------------------------- Пример минимального теста ---------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_allocate_write_read() -> SmfsResult<()> {
        let backend = InMemoryStorage::new();
        let saq = SaqNaive;
        let cipher = CipherNone;
        let mut mgr = SMFSQEManager::new(backend, saq, cipher);

        let id = mgr.allocate_block(BlockSize::B4KB, Some("episode:init".into()), None)?;
        mgr.write_range(id, 10, b"SRIS")?;
        let got = mgr.read_range(id, 10, 4)?;
        assert_eq!(&got, b"SRIS");
        Ok(())
    }
}
